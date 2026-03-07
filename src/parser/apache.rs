use anyhow::{anyhow, Result};
use regex::Regex;

// Bring the shared types from the parent module into scope.
// `super::` is the idiomatic way to reference a sibling or parent module
// without using the full crate path, keeping the import stable if the
// module is ever moved within the crate hierarchy.
use super::{LogRecord, ApacheRecord, Parser};

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
