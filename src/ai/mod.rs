pub mod anthropic;
#[cfg(test)]
pub mod mock;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::aggregator::LogSummary;

/// Severity of a triage issue. Serialises to/from lowercase JSON strings
/// ("critical", "warning", "info") matching the LLM response contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Critical => write!(f, "CRITICAL"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Info => write!(f, "INFO"),
        }
    }
}

/// Role of a message turn in a conversation.
/// Serialises to/from lowercase JSON strings ("user", "assistant") following
/// the Anthropic Messages API wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single triage issue identified by the AI layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub severity: Severity,
    /// One-line label, e.g. "High 5xx error rate"
    pub title: String,
    /// 1–2 sentences describing what is wrong
    pub explanation: String,
    /// Concrete next step for the engineer
    pub action: String,
    /// Indices into `LogSummary.top_errors` that support this issue.
    /// The LLM assigns these at generation time; the server uses them to fetch
    /// representative raw log lines without any post-hoc text matching.
    #[serde(default)]
    pub evidence_indices: Vec<usize>,
    /// Representative raw log lines supporting this issue.
    /// Populated by the server from `evidence_indices` — never sent to the model.
    #[serde(default)]
    pub evidence: Vec<String>,
}

/// The structured analysis produced by the AI layer for a log batch.
///
/// `issues` contains the ordered list of triage items (critical first).
/// `raw` holds the raw LLM response text when JSON parsing fails, so no
/// information is lost on a malformed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub issues: Vec<Issue>,
    /// Fallback populated only when the LLM response could not be parsed as JSON.
    pub raw: Option<String>,
}

/// A single turn in a chat conversation.
///
/// `Clone` is derived because the server appends to and passes history slices
/// without consuming the stored vec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// The single interface the server and CLI use for all AI interactions.
///
/// # Why `Send + Sync`
/// The trait is held behind `Box<dyn AnalysisEngine>` in the CLI and will be
/// held behind `Arc<dyn AnalysisEngine>` in the Axum server, shared across
/// Tokio tasks. `Send` allows the value to move between threads; `Sync` allows
/// a shared reference to cross thread boundaries.
///
/// # Why `#[async_trait]`
/// Native `async fn` in traits (stable since Rust 1.75) produces
/// RPITIT futures that are not object-safe: the compiler cannot build a vtable
/// for them. `#[async_trait]` rewrites each `async fn` to return
/// `Pin<Box<dyn Future + Send>>`, which IS object-safe and allows
/// `Box<dyn AnalysisEngine>` / `Arc<dyn AnalysisEngine>` to work.
/// The heap allocation per call is acceptable here — API latency dominates.
#[async_trait]
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
