use anyhow::Result;
use serde::Serialize;

// Declare the apache submodule. Rust will look for src/parser/apache.rs
// automatically — the file name is the module name by convention.
pub mod apache;

/// All parsed log lines are represented as typed enum variants — one per format.
/// Using an enum (rather than a trait object or HashMap) means match expressions
/// on LogRecord are exhaustive: the compiler forces callers to handle every format.
#[derive(Debug, Serialize)]
pub enum LogRecord {
    /// Apache Combined Log Format record.
    Apache(ApacheRecord),
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
}

/// The core parsing contract every format adapter must fulfil.
///
/// Taking `&str` (not `String`) avoids a heap allocation per line — callers
/// typically hold the line in a read buffer and we only borrow it during parsing.
pub trait Parser {
    fn parse(&self, line: &str) -> Result<LogRecord>;
}
