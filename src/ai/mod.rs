use anyhow::Result;
use serde::Serialize;

use crate::aggregator::LogSummary;

/// The plain-English analysis produced by the AI layer for a log batch.
///
/// Kept as a dedicated struct (rather than a bare String) so additional fields
/// (e.g. confidence score, structured recommendations) can be added later
/// without changing the AnalysisEngine trait signature.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    /// Plain-English summary of the log batch produced by the LLM.
    pub text: String,
}

/// A single turn in a chat conversation.
///
/// `role` follows the OpenAI / Anthropic convention ("user" | "assistant")
/// so the struct maps directly to API request bodies without transformation.
/// `Clone` is derived because the server will need to append to and pass
/// history slices without consuming the stored vec.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// Either "user" or "assistant".
    pub role: String,
    pub content: String,
}

/// The single interface the server and CLI use for all AI interactions.
///
/// # Why `Send + Sync`
/// The trait will be held behind `Arc<dyn AnalysisEngine>` in the Axum server
/// so it can be shared across async tasks on the Tokio thread pool. Both bounds
/// are required: `Send` to move across threads, `Sync` to share a reference.
///
/// # Why native async fn (not `async-trait`)
/// Rust 1.75+ stabilised `async fn` in traits. We use it here because:
/// - No extra crate dependency or macro expansion overhead.
/// - The compiler desugars each method to an RPITIT (return-position impl
///   trait in trait), which is zero-cost compared to `async-trait`'s
///   heap-allocated `Pin<Box<dyn Future>>`.
///
/// The trade-off: methods with native `async fn` are not yet usable through
/// `dyn AnalysisEngine` directly. When the server needs dynamic dispatch it
/// will use the `async-trait` crate or a hand-rolled `Box<dyn Future>` shim.
/// That decision is deferred to the implementation step (ai/anthropic.rs).
pub trait AnalysisEngine: Send + Sync {
    /// Produce a plain-English analysis of the aggregated log data.
    ///
    /// Accepts `&LogSummary` (not raw lines) to enforce the contract that the
    /// AI layer never receives raw log content — only the compact summary.
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult>;

    /// Continue a multi-turn conversation about the log data.
    ///
    /// `history` is a slice so callers can pass any contiguous sub-sequence
    /// of the full conversation without cloning the entire vec.
    async fn chat(&self, history: &[Message], question: &str) -> Result<String>;
}
