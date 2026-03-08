use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
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
use crate::parser::Parser;
use crate::store::{ResultStore, StoredResult};

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

/// All shared state passed to every Axum handler.
///
/// Both fields are `Clone`:
/// - `Arc<dyn AnalysisEngine>` clones cheaply (reference-counted pointer).
/// - `ResultStore` clones cheaply (`SqlitePool` is internally `Arc`-backed).
#[derive(Clone)]
struct AppState {
    engine: Arc<dyn AnalysisEngine>,
    store: ResultStore,
}

// ---------------------------------------------------------------------------
// Wire types — only used for HTTP request/response bodies
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SummaryQuery {
    file: String,
}

#[derive(Serialize)]
struct SummaryResponse {
    summary: LogSummary,
    analysis: AnalysisResult,
}

#[derive(Deserialize)]
struct ChatRequest {
    question: String,
    history: Vec<Message>,
    #[allow(dead_code)]
    summary: LogSummary,
}

#[derive(Serialize)]
struct ChatResponse {
    answer: String,
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
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

/// Constructs the Axum router with all routes and shared state.
///
/// Accepts a concrete `ResultStore` (not boxed) because `ResultStore` is a
/// concrete type whose `Clone` impl is cheap — no trait object needed.
pub fn build_router(engine: Box<dyn AnalysisEngine>, store: ResultStore) -> Router {
    let state = AppState {
        engine: Arc::from(engine),
        store,
    };

    Router::new()
        .route("/api/summary", get(summary_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/results", get(list_results_handler))
        .route("/api/results/:id", get(get_result_handler))
        .with_state(state)
        .fallback_service(ServeDir::new("frontend").append_index_html_on_directories(true))
        .layer(CorsLayer::permissive())
}

/// Starts the Axum HTTP server, opening (or creating) the result store first.
pub async fn start_server(engine: Box<dyn AnalysisEngine>, host: &str, port: u16) -> Result<()> {
    let db_path = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:./log_lens.db".to_string());
    let store = ResultStore::new(&db_path).await?;

    let app = build_router(engine, store);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn summary_handler(
    State(state): State<AppState>,
    Query(params): Query<SummaryQuery>,
) -> Result<Json<SummaryResponse>, AppError> {
    let summary = parse_file_to_summary(&params.file).await?;
    let analysis = state.engine.analyse(&summary).await?;

    // Persist the result; a store failure does not fail the HTTP response.
    if let Err(e) = state.store.save(&params.file, &summary, &analysis).await {
        eprintln!("[warn] store.save failed: {e}");
    }

    Ok(Json(SummaryResponse { summary, analysis }))
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(body): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let answer = state.engine.chat(&body.history, &body.question).await?;
    Ok(Json(ChatResponse { answer }))
}

/// Returns all stored results as a JSON array, newest first.
async fn list_results_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<StoredResult>>, AppError> {
    let results = state.store.list().await?;
    Ok(Json(results))
}

/// Returns a single stored result by ID, or 404 if not found.
async fn get_result_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Response, AppError> {
    match state.store.get(id).await? {
        Some(detail) => Ok(Json(detail).into_response()),
        None => Ok(StatusCode::NOT_FOUND.into_response()),
    }
}

// ---------------------------------------------------------------------------
// Shared pipeline helper
// ---------------------------------------------------------------------------

async fn parse_file_to_summary(path: &str) -> Result<LogSummary> {
    let file = File::open(path)?;

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

    let apache_parser = ApacheParser::new()?;
    let tier2 = &sample[..sample.len().min(5)];
    let hits = tier2.iter().filter(|l| apache_parser.parse(l).is_ok()).count();

    let parser: Box<dyn Parser> = if hits * 2 >= tier2.len() {
        Box::new(apache_parser)
    } else {
        Box::new(AiInferredParser::new(&sample).await?)
    };

    let records = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
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
    use tower::ServiceExt;

    async fn body_bytes(body: axum::body::Body) -> bytes::Bytes {
        use http_body_util::BodyExt;
        body.collect().await.expect("body must be readable").to_bytes()
    }

    /// Creates an in-memory SQLite store suitable for tests.
    /// `:memory:` is per-connection, so each test gets a clean slate.
    async fn test_store() -> ResultStore {
        ResultStore::new("sqlite::memory:")
            .await
            .expect("in-memory store must succeed")
    }

    #[tokio::test]
    async fn get_summary_returns_200_with_valid_json() {
        let app = build_router(Box::new(MockEngine), test_store().await);

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

        let issues = json["analysis"]["issues"]
            .as_array()
            .expect("analysis.issues must be an array");
        assert!(!issues.is_empty(), "mock must return at least one issue");
        assert_eq!(issues[0]["severity"].as_str().unwrap(), "critical");
    }

    #[tokio::test]
    async fn post_chat_returns_200_with_answer() {
        let app = build_router(Box::new(MockEngine), test_store().await);

        let body = serde_json::json!({
            "question": "What is the error rate?",
            "history": [],
            "summary": { "total": 6, "error_rate": 0.0, "status_counts": { "200": 6 } }
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
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

        let answer = json["answer"].as_str().unwrap();
        assert!(!answer.is_empty(), "answer must not be empty");
    }

    #[tokio::test]
    async fn get_results_returns_empty_array_initially() {
        let app = build_router(Box::new(MockEngine), test_store().await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/results")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("handler must not panic");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body_bytes(response.into_body()).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json, serde_json::json!([]));
    }

    #[tokio::test]
    async fn get_result_by_id_returns_404_for_missing() {
        let app = build_router(Box::new(MockEngine), test_store().await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/results/9999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("handler must not panic");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
