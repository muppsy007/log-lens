use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{anyhow, Result};
use regex::Regex;
use sha2::{Digest, Sha256};

use super::{InferredRecord, LogRecord, Parser};

const MODEL: &str = "claude-sonnet-4-20250514";
const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Keep the regex short so the response stays well within max_tokens.
const MAX_TOKENS: u32 = 512;

// Static regex patterns used to compute structural shape.
// `LazyLock` compiles them once on first access rather than on every call;
// regex compilation is expensive (~microseconds) and would be a measurable
// cost when processing millions of lines.
static RE_TS_APACHE: LazyLock<Regex> = LazyLock::new(|| {
    // Apache Combined Log Format timestamp: 07/Mar/2026:09:12:03 -0500
    Regex::new(r"\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}").unwrap()
});

static RE_TS_ISO: LazyLock<Regex> = LazyLock::new(|| {
    // ISO 8601 / RFC 3339 timestamps with optional fractional seconds and timezone:
    // 2026-03-07T09:12:03, 2026-03-07 09:12:03, 2026-03-07T09:12:03.456+12:00
    Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
        .unwrap()
});

static RE_TS_SYSLOG: LazyLock<Regex> = LazyLock::new(|| {
    // Syslog timestamp: Mar  7 09:12:03
    Regex::new(r"[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}").unwrap()
});

static RE_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

static RE_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z]+").unwrap());

/// Parses log lines using an LLM-inferred regex.
///
/// `new()` is async because constructing the parser may involve an HTTP call
/// to the Anthropic API on a cache miss. Once built, `parse()` is fully
/// synchronous — the regex is compiled and ready.
pub struct AiInferredParser {
    re: Regex,
}

impl AiInferredParser {
    /// Constructs a parser for the given sample lines.
    ///
    /// On cache hit the regex is loaded from disk and compiled with no API call.
    /// On cache miss the LLM is queried once, the regex is cached, then compiled.
    pub async fn new(sample_lines: &[&str]) -> Result<Self> {
        let shape = structural_shape(sample_lines);
        let cache_key = sha256_hex(&shape);

        let cache = load_cache()?;

        let regex_str = if let Some(cached) = cache.get(&cache_key) {
            cached.clone()
        } else {
            let regex_str = infer_regex_from_llm(sample_lines).await?;
            // Re-load to avoid losing any writes that happened while we were
            // awaiting the LLM response (shouldn't matter for a single-user
            // CLI, but is a good habit for any cache write path).
            let mut fresh_cache = load_cache()?;
            fresh_cache.insert(cache_key, regex_str.clone());
            save_cache(&fresh_cache)?;
            regex_str
        };

        // Fail fast here rather than on the first `parse()` call — a bad
        // regex from the LLM surfaces immediately at construction time.
        let re = Regex::new(&regex_str)
            .map_err(|e| anyhow!("LLM-returned regex failed to compile: {e}\nRegex was: {regex_str}"))?;

        Ok(Self { re })
    }
}

impl Parser for AiInferredParser {
    fn parse(&self, line: &str) -> Result<LogRecord> {
        let caps = self
            .re
            .captures(line)
            .ok_or_else(|| anyhow!("line does not match inferred format: {:?}", line))?;

        // `capture_names()` iterates over all group names in pattern order.
        // It yields `Option<&str>` — `None` for unnamed/positional groups —
        // so `.flatten()` skips those and leaves only named capture group names.
        let mut fields = HashMap::new();
        for name in self.re.capture_names().flatten() {
            if let Some(m) = caps.name(name) {
                fields.insert(name.to_string(), m.as_str().to_string());
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
        // Join with newline so the shape captures inter-line structure too.
        .collect::<Vec<_>>()
        .join("\n")
}

fn shape_line(line: &str) -> String {
    // Timestamps must be replaced before numbers/words: "07/Mar/2026:09:12:03"
    // contains both digits and alphabetic chars; replacing timestamps first
    // prevents "Mar" from becoming "W" before the Apache pattern can match it.
    //
    // Intermediate placeholders use non-printable ASCII control characters
    // (below 0x20) so that RE_NUMBER (`\d+`) and RE_WORD (`[A-Za-z]+`) cannot
    // accidentally match and overwrite a token that was just written by an
    // earlier pass. The final `.replace()` calls swap them for the readable
    // single-letter tokens that form the structural key.
    let s = RE_TS_APACHE.replace_all(line, "\x01");
    let s = RE_TS_ISO.replace_all(&s, "\x01");
    let s = RE_TS_SYSLOG.replace_all(&s, "\x01");
    // Numbers after timestamps so digits already consumed by \x01 are not re-matched.
    let s = RE_NUMBER.replace_all(&s, "\x02");
    // Words last — at this point only punctuation, spaces, and the \x01/\x02 tokens remain.
    let s = RE_WORD.replace_all(&s, "\x03");
    // `to_string()` on the final `Cow<str>` materialises an owned String before
    // the chained `.replace()` calls (which operate on `&str`, not `Cow`).
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
    // `finalize()` returns `GenericArray<u8, U32>`; collect each byte as a
    // zero-padded 2-char hex string so the output is always 64 characters.
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

/// Reads the cache file at `path`, returning an empty map if the file does not exist.
/// Separated from `load_cache` so tests can provide a temp path without env-var tricks.
pub(crate) fn load_cache_from(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(path)?;
    // Propagate a JSON parse error rather than silently returning an empty map,
    // so a corrupted cache file is visible rather than causing repeated LLM calls.
    Ok(serde_json::from_str(&content)?)
}

fn load_cache() -> Result<HashMap<String, String>> {
    load_cache_from(&cache_path()?)
}

/// Writes the cache to `path`, creating parent directories as needed.
/// Separated from `save_cache` for the same testability reason as `load_cache_from`.
pub(crate) fn save_cache_to(path: &Path, cache: &HashMap<String, String>) -> Result<()> {
    // `parent()` returns None only if `path` is the filesystem root, which
    // cannot happen for our `~/.cache/…` path.
    if let Some(dir) = path.parent() {
        // `create_dir_all` is idempotent — safe to call even if the dir exists.
        fs::create_dir_all(dir)?;
    }
    let content = serde_json::to_string_pretty(cache)?;
    fs::write(path, content)?;
    Ok(())
}

fn save_cache(cache: &HashMap<String, String>) -> Result<()> {
    save_cache_to(&cache_path()?, cache)
}

// ---------------------------------------------------------------------------
// LLM call
// ---------------------------------------------------------------------------

async fn infer_regex_from_llm(sample_lines: &[&str]) -> Result<String> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow!("ANTHROPIC_API_KEY is not set"))?;

    let sample_text = sample_lines.iter().take(20).cloned().collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Here are sample lines from an unknown log format. Return ONLY a Rust \
         regex string with named capture groups that parses these lines. \
         No explanation, no markdown, just the raw regex.\n\n{sample_text}"
    );

    // Build a fresh client — `infer_regex_from_llm` is called at most once per
    // novel format (result is cached), so connection-pool reuse is not needed here.
    let client = reqwest::Client::new();

    // `serde_json::json!` builds the request body inline without needing
    // intermediate serialisable structs — appropriate for a one-off call.
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

    // Strip any markdown fences the model might add despite the prompt.
    // Trim leading/trailing whitespace, then strip any ``` block delimiters
    // and an optional language tag (e.g. "regex" or "rust").
    let cleaned = raw
        .trim()
        .trim_start_matches("```")
        .trim_start_matches("regex")
        .trim_start_matches("rust")
        .trim_end_matches("```")
        .trim();

    Ok(cleaned.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A short, predictable line used to verify structural shape substitutions.
    // Chosen to exercise all three substitution types (T, N, W) clearly.
    const SHAPE_LINE: &str = "2026-03-07T09:12:03 ERROR 127.0.0.1 404";

    #[test]
    fn structural_shape_substitutes_correctly() {
        let shape = structural_shape(&[SHAPE_LINE]);
        // 2026-03-07T09:12:03 → T (ISO timestamp)
        // ERROR               → W
        // 127, 0, 0, 1        → N each (numbers), dots preserved
        // 404                 → N
        assert_eq!(shape, "T W N.N.N.N N");
    }

    #[test]
    fn cache_round_trip_preserves_regex() {
        // Use a unique temp path so parallel test runs don't collide.
        // `std::process::id()` gives the PID, which is unique per process.
        let tmp = std::env::temp_dir()
            .join(format!("log-lens-cache-test-{}.json", std::process::id()));

        let mut cache = HashMap::new();
        // A minimal named-group regex — representative of what the LLM returns.
        cache.insert("deadbeef".to_string(), r"(?P<msg>.+)".to_string());

        save_cache_to(&tmp, &cache).expect("save must succeed");
        let loaded = load_cache_from(&tmp).expect("load must succeed");

        assert_eq!(
            loaded.get("deadbeef").map(String::as_str),
            Some(r"(?P<msg>.+)")
        );

        // Clean up — ignore errors (e.g. file already removed by another run).
        let _ = std::fs::remove_file(&tmp);
    }

    // Integration tests below are marked `#[ignore]` because they make real
    // HTTP requests to the Anthropic API and require ANTHROPIC_API_KEY to be
    // set. Running them in CI without a key would cause false failures.
    //
    // Run manually:
    //   ANTHROPIC_API_KEY=sk-ant-... cargo test -- --ignored

    #[tokio::test]
    #[ignore]
    async fn infers_and_parses_unknown_format() {
        let lines: &[&str] = &[
            "2026-03-07T09:12:03 ERROR 127.0.0.1 404",
            "2026-03-07T09:13:00 INFO  10.0.0.1  200",
        ];
        let parser = AiInferredParser::new(lines)
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
