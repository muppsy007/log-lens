use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{InferredRecord, LogRecord, Parser};
use crate::ai::anthropic::{API_URL, ANTHROPIC_VERSION, MAX_TOKENS, MODEL};

// Static regex patterns used to compute structural shape.
static RE_TS_APACHE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}").unwrap()
});

static RE_TS_ISO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
        .unwrap()
});

static RE_TS_SYSLOG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}").unwrap()
});

static RE_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

static RE_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z]+").unwrap());

// ---------------------------------------------------------------------------
// Parse strategy
// ---------------------------------------------------------------------------

/// Full parsing strategy returned by the LLM and stored in the cache.
///
/// Separating the regex from the context-field metadata means the parser can
/// apply a two-phase approach: outer regex captures all fields (including any
/// embedded JSON blob as a raw string), then optionally parses that blob to
/// merge its keys into the record's field map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ParseStrategy {
    outer_regex: String,
    context_field: Option<String>,
    parse_context_as_json: bool,
}

// ---------------------------------------------------------------------------
// Parser struct and impl
// ---------------------------------------------------------------------------

/// Parses log lines using an LLM-inferred two-phase strategy.
///
/// Phase 1: the outer regex captures all fields, treating embedded JSON blobs
///          as opaque strings via a permissive capture group.
/// Phase 2 (optional): parse the captured blob as JSON and merge its keys
///          into the record field map, never failing the line on blob errors.
pub struct AiInferredParser {
    re: Regex,
    context_field: Option<String>,
    parse_context_as_json: bool,
}

impl AiInferredParser {
    /// Constructs a parser for the given sample lines.
    ///
    /// Returns `(parser, cache_hit)`. `cache_hit` is `true` when the strategy
    /// was loaded from disk with no API call; `false` when the LLM was queried.
    /// Callers use the flag to emit an accurate progress message rather than
    /// guessing before the call whether the cache will hit.
    pub async fn new(sample_lines: &[&str]) -> Result<(Self, bool)> {
        let shape = structural_shape(sample_lines);
        let cache_key = sha256_hex(&shape);

        let mut cache = load_cache()?;

        let client = reqwest::Client::new();

        let (strategy, cache_hit) = if let Some(cached) = cache.get(&cache_key) {
            (cached.clone(), true)
        } else {
            let strategy = infer_strategy_from_llm(sample_lines, &client).await?;
            cache.insert(cache_key, strategy.clone());
            save_cache(&cache)?;
            (strategy, false)
        };

        let re = Regex::new(&strategy.outer_regex).map_err(|e| {
            anyhow!(
                "LLM-returned regex failed to compile: {e}\nRegex was: {}",
                strategy.outer_regex
            )
        })?;

        Ok((Self {
            re,
            context_field: strategy.context_field,
            parse_context_as_json: strategy.parse_context_as_json,
        }, cache_hit))
    }
}

impl Parser for AiInferredParser {
    fn parse(&self, line: &str) -> Result<LogRecord> {
        let caps = self
            .re
            .captures(line)
            .ok_or_else(|| anyhow!("line does not match inferred format: {:?}", line))?;

        let mut fields = HashMap::new();
        for name in self.re.capture_names().flatten() {
            if let Some(m) = caps.name(name) {
                fields.insert(name.to_string(), m.as_str().to_string());
            }
        }

        // Phase 2: if the strategy identified a JSON blob field, attempt to
        // parse it and merge its keys. Failure is non-fatal — the raw string
        // value is kept and the line parse succeeds regardless.
        if self.parse_context_as_json {
            if let Some(ref ctx_field) = self.context_field {
                if let Some(raw_ctx) = fields.get(ctx_field).cloned() {
                    if let Ok(obj) =
                        serde_json::from_str::<HashMap<String, serde_json::Value>>(&raw_ctx)
                    {
                        for (k, v) in obj {
                            let s = match v {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            };
                            fields.insert(k, s);
                        }
                    }
                }
            }
        }

        Ok(LogRecord::Inferred(InferredRecord { fields, raw: line.to_string() }))
    }
}

// ---------------------------------------------------------------------------
// Structural shape
// ---------------------------------------------------------------------------

/// Reduces sample lines to their structural shape by normalising values
/// while preserving delimiter/punctuation characters.
///
/// `pub` so tests in this module can call it directly.
pub fn structural_shape(lines: &[&str]) -> String {
    lines
        .iter()
        .map(|line| shape_line(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn shape_line(line: &str) -> String {
    let s = RE_TS_APACHE.replace_all(line, "\x01");
    let s = RE_TS_ISO.replace_all(&s, "\x01");
    let s = RE_TS_SYSLOG.replace_all(&s, "\x01");
    let s = RE_NUMBER.replace_all(&s, "\x02");
    let s = RE_WORD.replace_all(&s, "\x03");
    s.to_string()
        .replace('\x01', "T")
        .replace('\x02', "N")
        .replace('\x03', "W")
}

// ---------------------------------------------------------------------------
// SHA-256 cache key
// ---------------------------------------------------------------------------

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Disk cache helpers
// ---------------------------------------------------------------------------

fn cache_path() -> Result<PathBuf> {
    let home =
        env::var("HOME").map_err(|_| anyhow!("HOME environment variable is not set"))?;
    Ok(PathBuf::from(home).join(".cache/log-lens/schemas.json"))
}

/// Reads the cache file at `path`, returning an empty map if the file does not
/// exist or cannot be parsed (e.g. old single-regex format on disk).
pub(crate) fn load_cache_from(path: &Path) -> Result<HashMap<String, ParseStrategy>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(path)?;
    // Return empty on parse failure so a stale/old-format cache is silently
    // invalidated rather than causing an error on every run.
    Ok(serde_json::from_str(&content).unwrap_or_default())
}

fn load_cache() -> Result<HashMap<String, ParseStrategy>> {
    load_cache_from(&cache_path()?)
}

/// Writes the cache to `path`, creating parent directories as needed.
pub(crate) fn save_cache_to(path: &Path, cache: &HashMap<String, ParseStrategy>) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let content = serde_json::to_string_pretty(cache)?;
    fs::write(path, content)?;
    Ok(())
}

fn save_cache(cache: &HashMap<String, ParseStrategy>) -> Result<()> {
    save_cache_to(&cache_path()?, cache)
}

// ---------------------------------------------------------------------------
// LLM call
// ---------------------------------------------------------------------------

async fn infer_strategy_from_llm(
    sample_lines: &[&str],
    client: &reqwest::Client,
) -> Result<ParseStrategy> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow!("ANTHROPIC_API_KEY is not set"))?;

    let sample_text = sample_lines.iter().take(20).cloned().collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Analyse these log lines and return ONLY a JSON object with no markdown:\n\
         {{\n\
           \"outer_regex\": \"a Rust regex with named capture groups that captures all \
         fields, using a permissive pattern like (?P<context>\\\\{{.*\\\\}}) for any \
         embedded JSON blob rather than trying to match its contents\",\n\
           \"context_field\": \"name of the capture group that contains a JSON blob, \
         or null if none\",\n\
           \"parse_context_as_json\": true or false\n\
         }}\n\
         Sample lines:\n\
         {sample_text}"
    );

    let body = serde_json::json!({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}]
    });

    let response = client
        .post(API_URL)
        .header("x-api-key", &api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(anyhow!("Anthropic API returned {status}: {text}"));
    }

    let json: serde_json::Value = response.json().await?;

    let raw = json["content"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Anthropic API response contained no text content"))?;

    // Use a streaming deserializer starting from the first `{` so any
    // surrounding prose or markdown fences are ignored.
    let start = raw
        .find('{')
        .ok_or_else(|| anyhow!("LLM response contained no JSON object\nRaw: {raw}"))?;
    let mut de = serde_json::Deserializer::from_str(&raw[start..]);
    let strategy = ParseStrategy::deserialize(&mut de)
        .map_err(|e| anyhow!("Failed to parse LLM strategy response: {e}\nRaw: {raw}"))?;

    Ok(strategy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE_LINE: &str = "2026-03-07T09:12:03 ERROR 127.0.0.1 404";

    #[test]
    fn structural_shape_substitutes_correctly() {
        let shape = structural_shape(&[SHAPE_LINE]);
        assert_eq!(shape, "T W N.N.N.N N");
    }

    #[test]
    fn cache_round_trip_preserves_strategy() {
        let tmp = std::env::temp_dir()
            .join(format!("log-lens-cache-test-{}.json", std::process::id()));

        let mut cache = HashMap::new();
        cache.insert(
            "deadbeef".to_string(),
            ParseStrategy {
                outer_regex: r"(?P<msg>.+)".to_string(),
                context_field: Some("context".to_string()),
                parse_context_as_json: true,
            },
        );

        save_cache_to(&tmp, &cache).expect("save must succeed");
        let loaded = load_cache_from(&tmp).expect("load must succeed");

        let s = loaded.get("deadbeef").expect("key must exist");
        assert_eq!(s.outer_regex, r"(?P<msg>.+)");
        assert_eq!(s.context_field.as_deref(), Some("context"));
        assert!(s.parse_context_as_json);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn old_format_cache_returns_empty_not_error() {
        // A cache file written by the old single-regex format should be
        // silently discarded rather than causing a parse error.
        let tmp = std::env::temp_dir()
            .join(format!("log-lens-cache-old-{}.json", std::process::id()));
        std::fs::write(&tmp, r#"{"deadbeef": "(?P<msg>.+)"}"#)
            .expect("write must succeed");

        let loaded = load_cache_from(&tmp).expect("load must succeed");
        assert!(loaded.is_empty(), "old format should deserialize to empty map");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn context_json_merged_into_fields() {
        // Verify phase-2 parsing: a captured JSON blob's keys are merged into
        // the record fields without failing the parse.
        let strategy = ParseStrategy {
            outer_regex: r#"(?P<level>\w+): (?P<message>[^{]+)(?P<context>\{.*\})"#.to_string(),
            context_field: Some("context".to_string()),
            parse_context_as_json: true,
        };
        let re = Regex::new(&strategy.outer_regex).unwrap();
        let parser = AiInferredParser {
            re,
            context_field: strategy.context_field,
            parse_context_as_json: strategy.parse_context_as_json,
        };

        let line = r#"ERROR: something went wrong {"code": "42", "user": "alice"}"#;
        let record = parser.parse(line).expect("line must parse");
        match record {
            LogRecord::Inferred(r) => {
                assert_eq!(r.fields.get("level").map(String::as_str), Some("ERROR"));
                // Context JSON keys must be merged in.
                assert_eq!(r.fields.get("code").map(String::as_str), Some("42"));
                assert_eq!(r.fields.get("user").map(String::as_str), Some("alice"));
            }
            _ => panic!("expected LogRecord::Inferred"),
        }
    }

    #[test]
    fn malformed_context_blob_keeps_raw_string() {
        // If the context field is not valid JSON, the line must still parse
        // successfully and the raw string value must be retained.
        let strategy = ParseStrategy {
            outer_regex: r#"(?P<level>\w+): (?P<context>.*)"#.to_string(),
            context_field: Some("context".to_string()),
            parse_context_as_json: true,
        };
        let re = Regex::new(&strategy.outer_regex).unwrap();
        let parser = AiInferredParser {
            re,
            context_field: strategy.context_field,
            parse_context_as_json: strategy.parse_context_as_json,
        };

        let line = "ERROR: {not valid json at all";
        let record = parser.parse(line).expect("line must parse despite bad context blob");
        match record {
            LogRecord::Inferred(r) => {
                assert_eq!(
                    r.fields.get("context").map(String::as_str),
                    Some("{not valid json at all")
                );
            }
            _ => panic!("expected LogRecord::Inferred"),
        }
    }

    // Integration tests require ANTHROPIC_API_KEY — run with `cargo test -- --ignored`.

    #[tokio::test]
    #[ignore]
    async fn infers_and_parses_unknown_format() {
        let lines: &[&str] = &[
            "2026-03-07T09:12:03 ERROR 127.0.0.1 404",
            "2026-03-07T09:13:00 INFO  10.0.0.1  200",
        ];
        let (parser, _cache_hit) = AiInferredParser::new(lines)
            .await
            .expect("parser construction must succeed");

        let record = parser.parse(lines[0]).expect("line must parse");
        match record {
            LogRecord::Inferred(r) => {
                assert!(!r.fields.is_empty(), "inferred record must have fields");
            }
            _ => panic!("expected LogRecord::Inferred"),
        }
    }
}
