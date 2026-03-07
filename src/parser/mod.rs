use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;

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

/// Parses Apache Combined Log Format lines via a compiled regex.
///
/// The Regex is stored in the struct so it is compiled once at construction and
/// reused for every line. Regex compilation is orders of magnitude more expensive
/// than matching, so compiling inside `parse` would be a serious hot-path cost.
pub struct ApacheParser {
    re: Regex,
}

impl ApacheParser {
    pub fn new() -> Result<Self> {
        // Named capture groups map directly to ApacheRecord fields, making
        // the extraction code below self-documenting and refactor-safe.
        //
        // Pattern covers Apache Combined Log Format:
        //   %h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\"
        let re = Regex::new(
            r#"^(?P<ip>\S+) (?P<ident>\S+) (?P<auth>\S+) \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" (?P<status>\d{3}) (?P<bytes>\S+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)""#,
        )?;
        Ok(Self { re })
    }
}

impl Parser for ApacheParser {
    fn parse(&self, line: &str) -> Result<LogRecord> {
        // `captures` returns None for a non-matching line rather than an error,
        // so we convert None → anyhow::Error with context for callers.
        let caps = self
            .re
            .captures(line)
            .ok_or_else(|| anyhow!("line does not match Apache Combined Log Format: {:?}", line))?;

        // The bytes field is "-" when the response body is empty (e.g. 304 Not Modified).
        // Treating "-" as 0 keeps the type numeric without failing the parse.
        let bytes: u64 = match &caps["bytes"] {
            "-" => 0,
            s => s.parse()?,
        };

        Ok(LogRecord::Apache(ApacheRecord {
            ip: caps["ip"].to_string(),
            ident: caps["ident"].to_string(),
            auth: caps["auth"].to_string(),
            timestamp: caps["timestamp"].to_string(),
            method: caps["method"].to_string(),
            path: caps["path"].to_string(),
            // trim trailing space that the regex includes before the closing quote
            protocol: caps["protocol"].trim_end().to_string(),
            status: caps["status"].parse()?,
            bytes,
            referer: caps["referer"].to_string(),
            user_agent: caps["user_agent"].to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Taken verbatim from samples/apache-access.log line 1.
    const VALID_LINE: &str =
        r#"192.168.1.10 - - [07/Mar/2026:09:12:03 -0500] "GET /index.html HTTP/1.1" 200 5432 "-" "Mozilla/5.0""#;

    #[test]
    fn parses_valid_apache_line() {
        let parser = ApacheParser::new().expect("regex must compile");
        let record = parser.parse(VALID_LINE).expect("valid line must parse");
        match record {
            LogRecord::Apache(r) => {
                assert_eq!(r.ip, "192.168.1.10");
                assert_eq!(r.method, "GET");
                assert_eq!(r.path, "/index.html");
                assert_eq!(r.protocol, "HTTP/1.1");
                assert_eq!(r.status, 200);
                assert_eq!(r.bytes, 5432);
                assert_eq!(r.timestamp, "07/Mar/2026:09:12:03 -0500");
            }
        }
    }

    #[test]
    fn rejects_malformed_line() {
        let parser = ApacheParser::new().expect("regex must compile");
        // Structurally invalid — missing every field after the IP.
        let result = parser.parse("not a log line at all %%garbage");
        assert!(result.is_err());
    }
}
