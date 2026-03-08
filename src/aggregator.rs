use std::collections::HashMap;

use serde::Serialize;

// Pull in the types we aggregate over. `crate::` is used (rather than `super::`)
// because aggregator is a top-level module; it has no parent module to `super::` into.
use crate::parser::LogRecord;

/// The statistical summary produced from a batch of parsed log records.
/// Kept as a plain data struct (not an iterator/stream) so it can be serialised
/// to JSON, stored, or handed to the AI layer without further transformation.
#[derive(Debug, Serialize)]
pub struct LogSummary {
    /// Total number of records in the batch.
    pub total: u64,
    /// Fraction of records that returned a 5xx (server error) status code.
    /// Defined as 5xx-only — client errors (4xx) are tracked in status_counts
    /// but excluded from error_rate so operational alerting stays focused on
    /// server-side failures rather than normal 404/401 traffic.
    pub error_rate: f64,
    /// Count of records grouped by HTTP status code.
    /// HashMap<u16, u64>: u16 covers the full 100–599 status range; u64 for
    /// counts avoids overflow on high-volume log files.
    pub status_counts: HashMap<u16, u64>,
}

/// Aggregates a batch of parsed log records into a `LogSummary`.
///
/// Accepts ownership of the `Vec` so callers make the move/clone decision
/// explicitly; we never need to read the records again after aggregation.
pub fn aggregate(records: Vec<LogRecord>) -> LogSummary {
    // Cast to u64 up-front so arithmetic below stays in u64 throughout.
    let total = records.len() as u64;

    let mut status_counts: HashMap<u16, u64> = HashMap::new();

    for record in &records {
        match record {
            LogRecord::Apache(r) => {
                // `entry().or_insert(0)` is the idiomatic HashMap increment:
                // inserts 0 on first sight, then the `+= 1` updates in place.
                *status_counts.entry(r.status).or_insert(0) += 1;
            }
            LogRecord::Inferred(r) => {
                // Inferred records carry free-form fields; try common status
                // field names in priority order. `find_map` short-circuits on
                // the first name that exists in the map AND parses as u16.
                let status = ["status", "status_code", "code", "http_status"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                    .and_then(|v| v.parse::<u16>().ok());
                if let Some(s) = status {
                    *status_counts.entry(s).or_insert(0) += 1;
                }
            }
        }
    }

    // Sum counts for all 5xx status codes.
    // HashMap::iter yields `(&K, &V)` tuples; the closure receives a reference
    // to that tuple, giving `&(&u16, &u64)`. The outer `&` in `&(status, count)`
    // destructures the reference-to-tuple so `status` and `count` are `&u16`/`&u64`,
    // which we then dereference with `*` for arithmetic.
    let error_count: u64 = status_counts
        .iter()
        .filter(|&(status, _)| *status >= 500)
        .map(|(_, count)| count)
        .sum();

    // Guard against division by zero on an empty input.
    let error_rate = if total == 0 {
        0.0
    } else {
        error_count as f64 / total as f64
    };

    LogSummary {
        total,
        error_rate,
        status_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{ApacheRecord, LogRecord};

    /// Build a minimal ApacheRecord with only the status field set meaningfully.
    /// All other fields are valid but arbitrary — aggregation only needs status.
    fn apache_record(status: u16) -> LogRecord {
        LogRecord::Apache(ApacheRecord {
            ip: "127.0.0.1".to_string(),
            ident: "-".to_string(),
            auth: "-".to_string(),
            timestamp: "01/Jan/2026:00:00:00 +0000".to_string(),
            method: "GET".to_string(),
            path: "/".to_string(),
            protocol: "HTTP/1.1".to_string(),
            status,
            bytes: 512,
            referer: "-".to_string(),
            user_agent: "test-agent".to_string(),
        })
    }

    #[test]
    fn error_rate_counts_only_5xx() {
        // 2× 200, 1× 404, 1× 500 → error_rate = 1/4 = 0.25
        let records = vec![
            apache_record(200),
            apache_record(200),
            apache_record(404),
            apache_record(500),
        ];
        let summary = aggregate(records);

        assert_eq!(summary.total, 4);
        // Use an epsilon comparison — f64 equality is unreliable for exact fractions.
        assert!((summary.error_rate - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn status_counts_are_correct() {
        let records = vec![
            apache_record(200),
            apache_record(200),
            apache_record(404),
            apache_record(500),
        ];
        let summary = aggregate(records);

        assert_eq!(*summary.status_counts.get(&200).unwrap(), 2);
        assert_eq!(*summary.status_counts.get(&404).unwrap(), 1);
        assert_eq!(*summary.status_counts.get(&500).unwrap(), 1);
    }

    #[test]
    fn empty_input_returns_zero_error_rate() {
        // Guard: dividing by zero must not panic.
        let summary = aggregate(vec![]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.error_rate, 0.0);
    }
}
