use std::convert::Infallible;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::aggregator::{AggregatorOutput, LogSummary};
use crate::ai::{AnalysisEngine, AnalysisResult, Message};
use crate::pipeline;
use crate::store::{ResultStore, StoredResult};

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn AnalysisEngine>,
    store: ResultStore,
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SummaryQuery {
    file: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    question: String,
    history: Vec<Message>,
}

#[derive(Serialize)]
struct ChatResponse {
    answer: String,
}

// ---------------------------------------------------------------------------
// SSE event helpers
// ---------------------------------------------------------------------------

/// Builds a progress event (stage + message) for SSE.
fn progress(stage: &str, message: impl Into<String>) -> Event {
    let data = serde_json::json!({ "stage": stage, "message": message.into() });
    Event::default()
        .event("progress")
        .data(data.to_string())
}

/// Builds the terminal "complete" event carrying the full result payload.
fn complete_event(summary: &LogSummary, analysis: &AnalysisResult) -> Event {
    let data = serde_json::json!({
        "stage": "complete",
        "summary": summary,
        "analysis": analysis,
    });
    Event::default()
        .event("complete")
        .data(data.to_string())
}

/// Builds an error event. After sending this the pipeline task exits.
fn error_event(message: impl Into<String>) -> Event {
    let data = serde_json::json!({ "stage": "error", "message": message.into() });
    Event::default()
        .event("error")
        .data(data.to_string())
}

// ---------------------------------------------------------------------------
// Error handling (used by chat/results handlers only)
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

/// Streams the full analysis pipeline as SSE progress events, ending with a
/// "complete" event that carries the full `SummaryResponse` payload.
async fn summary_handler(
    State(state): State<AppState>,
    Query(params): Query<SummaryQuery>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Event>(16);
    let stream = ReceiverStream::new(rx).map(Ok::<Event, Infallible>);

    tokio::spawn(async move {
        macro_rules! send {
            ($evt:expr) => {
                // Ignore send errors — client may have disconnected.
                let _ = tx.send($evt).await;
            };
        }

        let out = match parse_file_to_aggregator_output(&params.file, &tx).await {
            Ok(o) => o,
            Err(e) => {
                send!(error_event(e.to_string()));
                return;
            }
        };

        let summary = out.summary;

        // Analysing
        send!(progress("analysing", "Sending summary to LLM for analysis"));

        let mut analysis = match state.engine.analyse(&summary).await {
            Ok(a) => a,
            Err(e) => {
                send!(error_event(e.to_string()));
                return;
            }
        };

        // Join evidence to issues using the indices the LLM assigned.
        //
        // The LLM includes `evidence_indices` in each issue: zero-based indices
        // into `summary.top_errors`. We look up the corresponding raw log lines
        // from `out.evidence` (keyed by the same pattern string used in top_errors)
        // and attach up to 3 lines per referenced entry.
        {
            // Build a lookup from top_errors pattern → sample lines.
            use std::collections::HashMap;
            let line_map: HashMap<&str, &[String]> = out
                .evidence
                .iter()
                .map(|ev| (ev.pattern.as_str(), ev.sample_lines.as_slice()))
                .collect();

            for issue in &mut analysis.issues {
                for &idx in &issue.evidence_indices {
                    if let Some(detail) = summary.top_errors.get(idx) {
                        if let Some(lines) = line_map.get(detail.message.as_str()) {
                            issue.evidence.extend(lines.iter().take(3).cloned());
                        }
                    }
                }
            }
        }

        if let Err(e) = state.store.save(&params.file, &summary, &analysis).await {
            eprintln!("[warn] store.save failed: {e}");
        }

        // Complete
        send!(complete_event(&summary, &analysis));
        // tx dropped here → stream closes
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(body): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let answer = state.engine.chat(&body.history, &body.question).await?;
    Ok(Json(ChatResponse { answer }))
}

async fn list_results_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<StoredResult>>, AppError> {
    let results = state.store.list().await?;
    Ok(Json(results))
}

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
// Pipeline helper — emits progress events at each stage
// ---------------------------------------------------------------------------

async fn parse_file_to_aggregator_output(
    path: &str,
    tx: &mpsc::Sender<Event>,
) -> Result<AggregatorOutput> {
    // Forward each pipeline stage as an SSE progress event.
    // `try_send` is sync and non-blocking; with channel capacity 16 and at
    // most ~6 progress events it will not fail in practice.
    let (out, _skipped) = pipeline::parse_and_aggregate(path, |stage, msg| {
        let _ = tx.try_send(progress(stage, msg));
    })
    .await?;
    Ok(out)
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

    async fn test_store() -> ResultStore {
        ResultStore::new("sqlite::memory:")
            .await
            .expect("in-memory store must succeed")
    }

    /// Extracts the JSON payload from the "complete" SSE event in an SSE body.
    ///
    /// SSE frames are separated by blank lines. Each frame has one or more
    /// `field: value` lines. We find the frame whose `event:` line is
    /// "complete" and return the `data:` value.
    fn extract_complete_event(body: &str) -> serde_json::Value {
        // Split into frames on blank lines.
        for frame in body.split("\n\n") {
            let mut event_type = "";
            let mut data = "";
            for line in frame.lines() {
                if let Some(v) = line.strip_prefix("event:") {
                    event_type = v.trim();
                } else if let Some(v) = line.strip_prefix("data:") {
                    data = v.trim();
                }
            }
            if event_type == "complete" && !data.is_empty() {
                return serde_json::from_str(data).expect("complete event data must be valid JSON");
            }
        }
        panic!("no 'complete' event found in SSE body:\n{body}");
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
        let body_str = std::str::from_utf8(&bytes).expect("body must be UTF-8");
        let json = extract_complete_event(body_str);

        assert!(json.get("summary").is_some(), "complete event must have 'summary'");
        assert!(json.get("analysis").is_some(), "complete event must have 'analysis'");

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
