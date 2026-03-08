use anyhow::Result;

use super::{AnalysisEngine, AnalysisResult, Message};
use crate::aggregator::LogSummary;

/// A test double for `AnalysisEngine` that returns canned responses.
///
/// `MockEngine` exists so tests and local development can exercise the full
/// request pipeline (CLI → aggregator → AI layer → output) without making
/// real API calls or requiring a valid `ANTHROPIC_API_KEY`.
///
/// The struct has no fields because it holds no state — every call returns
/// the same hardcoded response regardless of input.
pub struct MockEngine;

impl AnalysisEngine for MockEngine {
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult> {
        // We reference `summary` here so the compiler does not warn about the
        // unused parameter; the format string interpolates the total count to
        // make the canned response loosely reflect the input.
        Ok(AnalysisResult {
            text: format!(
                "Mock analysis: processed {} log records. \
                 No anomalies detected (this is a canned response).",
                summary.total
            ),
        })
    }

    async fn chat(&self, history: &[Message], question: &str) -> Result<String> {
        // `history.len()` is referenced to avoid an unused-variable warning and
        // to give the canned reply a hint of context-awareness in manual testing.
        Ok(format!(
            "Mock reply to '{}' (conversation length: {} turns).",
            question,
            history.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Constructs a minimal `LogSummary` for use in tests.
    /// Lives here rather than in a shared helper module because it is only
    /// needed by this test module at this stage of the build.
    fn stub_summary() -> LogSummary {
        LogSummary {
            total: 42,
            error_rate: 0.1,
            // `HashMap::new()` is fine here — the mock ignores status_counts.
            status_counts: HashMap::new(),
        }
    }

    // `#[tokio::test]` replaces `#[test]` for async test functions.
    // It spins up a single-threaded Tokio runtime for the test, matching the
    // minimal overhead needed for unit tests (no need for the full multi-thread pool).
    #[tokio::test]
    async fn analyse_returns_canned_result() {
        let engine = MockEngine;
        let result = engine
            .analyse(&stub_summary())
            .await
            .expect("mock must not fail");

        // Assert the total count was interpolated into the canned text.
        assert!(result.text.contains("42"));
        assert!(result.text.contains("Mock analysis"));
    }

    #[tokio::test]
    async fn chat_returns_canned_string() {
        let engine = MockEngine;
        let history = vec![Message {
            role: "user".to_string(),
            content: "What is the error rate?".to_string(),
        }];

        let reply = engine
            .chat(&history, "Tell me more.")
            .await
            .expect("mock must not fail");

        assert!(reply.contains("Tell me more."));
        // history has 1 turn — confirm the length is reflected in the reply.
        assert!(reply.contains("1 turns"));
    }
}
