use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::aggregator::{aggregate, LogSummary};
use crate::ai::{AnalysisEngine, AnalysisResult, Message};
use crate::parser::ai_infer::AiInferredParser;
use crate::parser::apache::ApacheParser;
// Import Parser trait so `.parse()` is in scope on both parser types.
use crate::parser::Parser;

// ---------------------------------------------------------------------------
// Wire types — only used for HTTP request/response bodies
// ---------------------------------------------------------------------------

/// Query parameters for GET /api/summary.
#[derive(Deserialize)]
struct SummaryQuery {
    file: String,
}

/// Response body for GET /api/summary.
#[derive(Serialize)]
struct SummaryResponse {
    summary: LogSummary,
    /// Structured triage analysis produced by the AI layer.
    analysis: AnalysisResult,
}

/// Request body for POST /api/chat.
#[derive(Deserialize)]
struct ChatRequest {
    question: String,
    history: Vec<Message>,
    /// The client echoes back the summary from a previous /api/summary call.
    /// Kept in the request so future implementations can inject it as context;
    /// the current handler uses only `question` and `history`.
    #[allow(dead_code)]
    summary: LogSummary,
}

/// Response body for POST /api/chat.
#[derive(Serialize)]
struct ChatResponse {
    answer: String,
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// A wrapper that converts any `anyhow::Error` into an HTTP 500 response.
///
/// Axum requires handlers to return `IntoResponse`. By implementing it on
/// `AppError`, we can use `?` in handlers for any function returning
/// `anyhow::Result` and get a clean error response without manual `.map_err`.
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError(e)
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Constructs the Axum router with all routes wired up.
///
/// Accepts `Box<dyn AnalysisEngine>` so callers (including tests) can inject
/// any implementation — `AnthropicEngine` in production, `MockEngine` in tests.
/// The Box is immediately wrapped in `Arc` because Axum state must be `Clone`,
/// and `Arc::from(Box<dyn T>)` is a zero-copy promotion via the `From` impl.
pub fn build_router(engine: Box<dyn AnalysisEngine>) -> Router {
    // Arc<dyn AnalysisEngine> satisfies Axum's Clone + Send + Sync + 'static
    // requirement for state. Wrapping here (not at the call site) keeps the
    // public API simple — callers don't need to know about Arc.
    let engine: Arc<dyn AnalysisEngine> = Arc::from(engine);

    Router::new()
        .route("/api/summary", get(summary_handler))
        .route("/api/chat", post(chat_handler))
        .with_state(engine)
        // `fallback_service` catches every request that does not match an API
        // route. `ServeDir` walks the `frontend/` directory relative to the
        // working directory (the project root when run via `cargo run`).
        // `append_index_html_on_directories` makes `GET /` serve `index.html`.
        .fallback_service(ServeDir::new("frontend").append_index_html_on_directories(true))
        // Permissive CORS so the single-file frontend can call the API from
        // any origin during local development without proxy configuration.
        .layer(CorsLayer::permissive())
}

/// Starts the Axum HTTP server.
///
/// Extracted from `main` so it can be driven by either the CLI flag or
/// future integration tests that want a real listening server.
pub async fn start_server(engine: Box<dyn AnalysisEngine>, host: &str, port: u16) -> Result<()> {
    let app = build_router(engine);
    let addr = format!("{host}:{port}");
    // `TcpListener::bind` resolves the address and claims the port.
    // Using `tokio::net::TcpListener` (not `std::net`) keeps this async.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Server listening on http://{addr}");
    // `axum::serve` drives the server until the process exits or an IO error occurs.
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn summary_handler(
    State(engine): State<Arc<dyn AnalysisEngine>>,
    Query(params): Query<SummaryQuery>,
) -> Result<Json<SummaryResponse>, AppError> {
    let summary = parse_file_to_summary(&params.file)
        .await
        .map_err(anyhow::Error::from)?;

    // Handlers must never call AnthropicEngine methods directly — only the trait.
    let analysis = engine.analyse(&summary).await?;

    Ok(Json(SummaryResponse { summary, analysis }))
}

async fn chat_handler(
    State(engine): State<Arc<dyn AnalysisEngine>>,
    Json(body): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let answer = engine.chat(&body.history, &body.question).await?;
    Ok(Json(ChatResponse { answer }))
}

// ---------------------------------------------------------------------------
// Shared pipeline helper
// ---------------------------------------------------------------------------

/// Reads a log file, detects its format, parses every line, and aggregates
/// the result into a `LogSummary`.
///
/// This is identical to the pipeline in `main.rs` but without terminal output,
/// so it is suitable for use inside an HTTP handler.
async fn parse_file_to_summary(path: &str) -> Result<LogSummary> {
    let file = File::open(path)?;

    // Collect all lines so we can sample them for format detection before
    // committing to a parser without reading the file twice.
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?;

    let sample: Vec<&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .take(20)
        .map(String::as_str)
        .collect();

    if sample.is_empty() {
        return Err(anyhow::anyhow!("No non-empty lines found in {:?}", path));
    }

    // Tier 2: Apache heuristic.
    let apache_parser = ApacheParser::new()?;
    let tier2 = &sample[..sample.len().min(5)];
    let hits = tier2.iter().filter(|l| apache_parser.parse(l).is_ok()).count();

    // Tier 3: LLM inference fallback if Apache does not match the majority.
    let parser: Box<dyn Parser> = if hits * 2 >= tier2.len() {
        Box::new(apache_parser)
    } else {
        Box::new(AiInferredParser::new(&sample).await?)
    };

    let records = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        // Silently drop lines that don't match the parser — same policy as the CLI.
        .filter_map(|l| parser.parse(l).ok())
        .collect();

    Ok(aggregate(records))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::mock::MockEngine;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    // `ServiceExt` adds `.oneshot()` to the Router, which sends a single
    // request through the full middleware stack and returns the response.
    // This avoids binding a real TCP port, keeping tests fast and isolated.
    use tower::ServiceExt;

    // Helper that reads a response body into raw bytes.
    async fn body_bytes(body: axum::body::Body) -> bytes::Bytes {
        use http_body_util::BodyExt;
        body.collect().await.expect("body must be readable").to_bytes()
    }

    #[tokio::test]
    async fn get_summary_returns_200_with_valid_json() {
        // MockEngine is injected so this test never touches the Anthropic API
        // and does not require ANTHROPIC_API_KEY to be set.
        let app = build_router(Box::new(MockEngine));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/summary?file=samples/apache-access.log")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("handler must not panic");

        assert_eq!(response.status(), StatusCode::OK);

        let bytes = body_bytes(response.into_body()).await;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response must be valid JSON");

        assert!(json.get("summary").is_some(), "response must have 'summary' field");
        assert!(json.get("analysis").is_some(), "response must have 'analysis' field");

        // The mock returns structured issues — verify the shape.
        let issues = json["analysis"]["issues"]
            .as_array()
            .expect("analysis.issues must be an array");
        assert!(!issues.is_empty(), "mock must return at least one issue");
        assert_eq!(
            issues[0]["severity"].as_str().unwrap(),
            "critical",
            "first issue must be critical"
        );
    }

    #[tokio::test]
    async fn post_chat_returns_200_with_answer() {
        let app = build_router(Box::new(MockEngine));

        // Construct a minimal but valid ChatRequest body.
        // The new `top_errors` and `top_slow_paths` fields have `#[serde(default)]`
        // so they are not required in the JSON body.
        let body = serde_json::json!({
            "question": "What is the error rate?",
            "history": [],
            "summary": {
                "total": 6,
                "error_rate": 0.0,
                // JSON object keys are strings; serde deserialises them to u16.
                "status_counts": { "200": 6 }
            }
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    // Content-Type must be set so Axum's Json extractor accepts the body.
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .expect("handler must not panic");

        assert_eq!(response.status(), StatusCode::OK);

        let bytes = body_bytes(response.into_body()).await;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response must be valid JSON");

        assert!(json.get("answer").is_some(), "response must have 'answer' field");
        let answer = json["answer"].as_str().unwrap();
        assert!(!answer.is_empty(), "answer must not be empty");
    }
}
