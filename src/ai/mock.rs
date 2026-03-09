use anyhow::Result;
use async_trait::async_trait;

use super::{AnalysisEngine, AnalysisResult, Issue, Message};
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

#[async_trait]
impl AnalysisEngine for MockEngine {
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult> {
        Ok(AnalysisResult {
            issues: vec![
                Issue {
                    severity: "critical".to_string(),
                    title: "Mock critical issue".to_string(),
                    explanation: format!(
                        "Mock analysis: processed {} log records with elevated error rate.",
                        summary.total
                    ),
                    action: "Investigate the root cause immediately.".to_string(),
                    evidence: vec![],
                },
                Issue {
                    severity: "warning".to_string(),
                    title: "Mock warning issue".to_string(),
                    explanation: "Some paths returned 4xx responses.".to_string(),
                    action: "Review client request patterns.".to_string(),
                    evidence: vec![],
                },
            ],
            raw: None,
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
    fn stub_summary() -> LogSummary {
        LogSummary {
            total: 42,
            error_rate: 0.1,
            status_counts: HashMap::new(),
            top_errors: vec![],
            top_slow_paths: vec![],
            suspicious_ips: vec![],
        }
    }

    #[tokio::test]
    async fn analyse_returns_structured_result() {
        let engine = MockEngine;
        let result = engine
            .analyse(&stub_summary())
            .await
            .expect("mock must not fail");

        assert!(!result.issues.is_empty(), "must return at least one issue");
        // The critical issue explanation interpolates the total count.
        assert!(result.issues[0].explanation.contains("42"));
        assert_eq!(result.issues[0].severity, "critical");
        assert_eq!(result.issues[1].severity, "warning");
        assert!(result.raw.is_none());
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
