use std::env;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{AnalysisEngine, AnalysisResult, Message};
use crate::summary::to_json;
use crate::aggregator::LogSummary;

/// Model used for all API calls. Defined as a constant so it appears in one
/// place and can be updated without hunting through method bodies.
const MODEL: &str = "claude-sonnet-4-20250514";

const API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic requires this header on every request to pin the API contract.
/// Without it the request is rejected, so it is a constant rather than a magic string.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Maximum tokens the model may produce in a single response.
/// 1024 is enough for a structured log analysis or a conversational reply
/// without risking runaway generation costs.
const MAX_TOKENS: u32 = 1024;

/// Calls the Anthropic Messages API using `reqwest`.
///
/// `client` is stored in the struct rather than created per-call because
/// `reqwest::Client` manages a connection pool internally — reusing it avoids
/// TCP handshake overhead on every API request.
pub struct AnthropicEngine {
    client: Client,
    api_key: String,
}

impl AnthropicEngine {
    /// Constructs a new engine, reading the API key from the environment.
    ///
    /// Fails fast at construction time so callers discover a missing key
    /// immediately (at startup) rather than on the first API call.
    pub fn new() -> Result<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| {
            anyhow!("ANTHROPIC_API_KEY environment variable is not set")
        })?;
        Ok(Self {
            // `Client::new()` is cheap — it does not open connections yet.
            client: Client::new(),
            api_key,
        })
    }

    /// Shared HTTP call used by both `analyse` and `chat`.
    ///
    /// Extracted into a private method to keep the trait implementations thin
    /// and to avoid duplicating error-handling logic.
    async fn call_api(&self, messages: Vec<ApiMessage>) -> Result<String> {
        let body = ApiRequest {
            // `to_string()` clones the constant into an owned String as required
            // by the serialisable struct — the constant itself cannot be moved.
            model: MODEL.to_string(),
            max_tokens: MAX_TOKENS,
            messages,
        };

        let response = self
            .client
            .post(API_URL)
            // Anthropic authenticates via a custom header, not Bearer auth.
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            // `.json()` both serialises `body` and sets Content-Type: application/json.
            .json(&body)
            .send()
            .await?;

        // Check the HTTP status before attempting to deserialise the body.
        // On error, capture the body text for a useful error message — consuming
        // the response here means we can't call `.json()` on it afterwards.
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Anthropic API returned {}: {}", status, text));
        }

        let api_response: ApiResponse = response.json().await?;

        // The Messages API can return multiple content blocks (e.g. tool_use,
        // thinking). We find the first text block and return its content.
        // `find_map` short-circuits on the first match, avoiding a full iteration.
        api_response
            .content
            .into_iter()
            .find_map(|block| if block.kind == "text" { block.text } else { None })
            .ok_or_else(|| anyhow!("Anthropic API response contained no text content block"))
    }
}

#[async_trait]
impl AnalysisEngine for AnthropicEngine {
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult> {
        // Serialise to compact JSON — only the aggregated summary is transmitted,
        // never raw log lines. `to_json` from summary.rs handles this.
        let summary_json = to_json(summary)?;

        let prompt = format!(
            "You are a senior engineer triaging a production incident.\n\
             Analyse this log summary and return ONLY a JSON object — no prose, no markdown, no explanation outside the JSON.\n\
             \n\
             Return this exact shape:\n\
             {{\"issues\": [{{\"severity\": \"critical|warning|info\", \"title\": \"short label\", \"explanation\": \"1-2 sentences on what is wrong\", \"action\": \"concrete next step\"}}]}}\n\
             \n\
             Order issues by severity: critical first. Be specific — reference actual counts, paths, and error messages from the data.\n\
             The `top_errors` field contains structured error entries with `message`, `level`, `count`, and optionally `file` and `line` fields.\n\
             When `file` and `line` are present, cite them in your explanation (e.g. \"Error in src/Foo.php:42\").\n\
             Do not invent data not present in the summary.\n\
             \n\
             Log summary:\n\
             {summary_json}\n\
             \n\
             Return ONLY valid JSON with no markdown code fences, no backticks, no explanation. Raw JSON only."
        );

        let messages = vec![ApiMessage {
            role: "user".to_string(),
            content: prompt,
        }];

        let text = self.call_api(messages).await?;

        // Parse the JSON response; fall back to `raw` if the model included
        // prose or markdown that prevents clean deserialization.
        let result = serde_json::from_str::<AnalysisResult>(&text)
            .unwrap_or_else(|_| AnalysisResult {
                issues: vec![],
                raw: Some(text),
            });

        Ok(result)
    }

    async fn chat(&self, history: &[Message], question: &str) -> Result<String> {
        // Map our domain `Message` type to the API wire format.
        // `.iter()` borrows; `.map()` projects each reference to an owned
        // `ApiMessage` so the vec owns all its data when passed to `call_api`.
        let mut messages: Vec<ApiMessage> = history
            .iter()
            .map(|m| ApiMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        // Append the current question as the final user turn.
        messages.push(ApiMessage {
            role: "user".to_string(),
            content: question.to_string(),
        });

        self.call_api(messages).await
    }
}

// ---------------------------------------------------------------------------
// Private types for API serialisation / deserialisation.
// These are intentionally not `pub` — nothing outside this module needs them.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    // Serde renames the JSON key "type" (a Rust keyword) to `kind`.
    #[serde(rename = "type")]
    kind: String,
    // `Option` because non-text block types (e.g. "tool_use") have no `text`
    // field. Using Option avoids a deserialisation error on those blocks.
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::aggregator::LogSummary;

    fn stub_summary() -> LogSummary {
        let mut status_counts = HashMap::new();
        status_counts.insert(200u16, 95u64);
        status_counts.insert(500u16, 5u64);
        LogSummary {
            total: 100,
            error_rate: 0.05,
            status_counts,
            top_errors: vec![],
            top_slow_paths: vec![],
            suspicious_ips: vec![],
        }
    }

    // These tests are marked `#[ignore]` because they make real HTTP calls to
    // the Anthropic API and require ANTHROPIC_API_KEY to be set in the
    // environment. Running them in CI without a key would cause false failures.
    //
    // Run manually with:
    //   ANTHROPIC_API_KEY=sk-ant-... cargo test -- --ignored

    #[tokio::test]
    #[ignore]
    async fn analyse_returns_nonempty_result() {
        let engine = AnthropicEngine::new().expect("ANTHROPIC_API_KEY must be set");
        let result = engine
            .analyse(&stub_summary())
            .await
            .expect("API call must succeed");
        assert!(
            !result.issues.is_empty() || result.raw.is_some(),
            "analysis must contain issues or a raw fallback"
        );
    }

    #[tokio::test]
    #[ignore]
    async fn chat_returns_nonempty_string() {
        let engine = AnthropicEngine::new().expect("ANTHROPIC_API_KEY must be set");
        let reply = engine
            .chat(&[], "What is the error rate?")
            .await
            .expect("API call must succeed");
        assert!(!reply.is_empty(), "chat reply must not be empty");
    }
}
