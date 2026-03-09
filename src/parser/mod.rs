use std::collections::HashMap;

use anyhow::Result;
use serde::Serialize;

// Declare the apache submodule. Rust will look for src/parser/apache.rs
// automatically — the file name is the module name by convention.
pub mod apache;

// Declare the AI-inferred parser submodule. Falls through to this when no
// built-in parser recognises the format.
pub mod ai_infer;

/// All parsed log lines are represented as typed enum variants — one per format.
/// Using an enum (rather than a trait object or HashMap) means match expressions
/// on LogRecord are exhaustive: the compiler forces callers to handle every format.
#[derive(Debug, Serialize)]
pub enum LogRecord {
    /// Apache Combined Log Format record.
    Apache(ApacheRecord),
    /// Record produced by the AI-inferred parser for unknown log formats.
    /// Fields are keyed by the named capture group names the LLM chose.
    Inferred(InferredRecord),
}

/// Typed fields extracted from a single Apache Combined Log Format line.
/// Every field is named after its semantic meaning so callsites read like
/// documentation rather than positional magic (e.g. `r.status` not `fields[8]`).
#[derive(Debug, Serialize)]
pub struct ApacheRecord {
    pub ip: String,
    /// RFC 1413 ident field — almost always "-" in practice; kept for fidelity.
    pub ident: String,
    /// Authenticated user — "-" when no HTTP auth is in use.
    pub auth: String,
    /// Raw timestamp string, e.g. "07/Mar/2026:09:12:03 -0500".
    /// Left as String here to avoid coupling the parsing layer to chrono;
    /// the aggregator can parse it into a DateTime if time-bucketing is needed.
    pub timestamp: String,
    pub method: String,
    pub path: String,
    pub protocol: String,
    /// HTTP status as u16 so the aggregator can bucket (2xx/4xx/5xx) without re-parsing.
    pub status: u16,
    /// Response body size in bytes. u64 handles large payloads without overflow.
    pub bytes: u64,
    pub referer: String,
    pub user_agent: String,
    /// The original unparsed log line, preserved so the aggregator can surface
    /// representative sample lines in triage output without a second file read.
    pub raw: String,
}

/// A record produced by the AI-inferred parser.
/// Field names match the named capture groups in the LLM-generated regex,
/// so the set of fields varies by log format and is not known at compile time.
/// HashMap<String, String> is the right type here — unlike ApacheRecord,
/// there is no fixed schema to encode in the type system.
#[derive(Debug, Serialize)]
pub struct InferredRecord {
    pub fields: HashMap<String, String>,
    /// The original unparsed log line, preserved for evidence sampling.
    pub raw: String,
}

/// The core parsing contract every format adapter must fulfil.
///
/// Taking `&str` (not `String`) avoids a heap allocation per line — callers
/// typically hold the line in a read buffer and we only borrow it during parsing.
pub trait Parser {
    fn parse(&self, line: &str) -> Result<LogRecord>;
}
