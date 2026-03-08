use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::parser::LogRecord;

/// The statistical summary produced from a batch of parsed log records.
/// Kept as a plain data struct (not an iterator/stream) so it can be serialised
/// to JSON, stored, or handed to the AI layer without further transformation.
#[derive(Debug, Serialize, Deserialize)]
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
    /// Top 10 error paths by frequency (path/message-prefix, count).
    /// Grouped by request path when available; otherwise by the first 60
    /// characters of the log message. Covers 4xx and 5xx responses.
    #[serde(default)]
    pub top_errors: Vec<(String, u32)>,
    /// Top 5 paths by p99 latency (path, p99_latency_ms).
    /// Empty when the parsed records contain no latency data.
    #[serde(default)]
    pub top_slow_paths: Vec<(String, f64)>,
}

/// Aggregates a batch of parsed log records into a `LogSummary`.
///
/// Accepts ownership of the `Vec` so callers make the move/clone decision
/// explicitly; we never need to read the records again after aggregation.
pub fn aggregate(records: Vec<LogRecord>) -> LogSummary {
    let total = records.len() as u64;

    let mut status_counts: HashMap<u16, u64> = HashMap::new();
    // (path or message-prefix) → error count
    let mut error_path_counts: HashMap<String, u32> = HashMap::new();
    // path → collected latency samples for p99 computation
    let mut latency_samples: HashMap<String, Vec<f64>> = HashMap::new();

    for record in &records {
        match record {
            LogRecord::Apache(r) => {
                *status_counts.entry(r.status).or_insert(0) += 1;

                // Group 4xx and 5xx errors by request path.
                if r.status >= 400 {
                    *error_path_counts.entry(r.path.clone()).or_insert(0) += 1;
                }

                // Apache Combined Log Format carries no latency field;
                // top_slow_paths will be empty for pure Apache logs.
            }
            LogRecord::Inferred(r) => {
                let status = ["status", "status_code", "code", "http_status"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                    .and_then(|v| v.parse::<u16>().ok());

                if let Some(s) = status {
                    *status_counts.entry(s).or_insert(0) += 1;

                    if s >= 400 {
                        // Prefer a dedicated path field; fall back to the first
                        // 60 characters of the first available field value.
                        let key = ["path", "url", "uri", "request_path"]
                            .iter()
                            .find_map(|&k| r.fields.get(k))
                            .cloned()
                            .unwrap_or_else(|| {
                                r.fields
                                    .values()
                                    .next()
                                    .map(|v| v.chars().take(60).collect())
                                    .unwrap_or_default()
                            });
                        *error_path_counts.entry(key).or_insert(0) += 1;
                    }
                }

                // Extract latency and group by path for p99 computation.
                let latency = ["response_time", "latency", "duration", "time_taken", "request_time"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                    .and_then(|v| v.parse::<f64>().ok());

                if let Some(lat) = latency {
                    let path = ["path", "url", "uri", "request_path"]
                        .iter()
                        .find_map(|&k| r.fields.get(k))
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    latency_samples.entry(path).or_default().push(lat);
                }
            }
        }
    }

    // Sum counts for all 5xx status codes.
    let error_count: u64 = status_counts
        .iter()
        .filter(|&(status, _)| *status >= 500)
        .map(|(_, count)| count)
        .sum();

    let error_rate = if total == 0 {
        0.0
    } else {
        error_count as f64 / total as f64
    };

    // Top 10 error paths, descending by count.
    let mut top_errors: Vec<(String, u32)> = error_path_counts.into_iter().collect();
    top_errors.sort_by(|a, b| b.1.cmp(&a.1));
    top_errors.truncate(10);

    // Top 5 paths by p99 latency, descending.
    let mut top_slow_paths: Vec<(String, f64)> = latency_samples
        .into_iter()
        .map(|(path, mut samples)| {
            // Sort ascending so index arithmetic gives p99.
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // p99 index: for N samples, take the value at floor(N * 99 / 100).
            // `saturating_sub` guards against the single-sample case (0 * 99/100 = 0).
            let p99_idx = (samples.len() * 99 / 100).min(samples.len() - 1);
            (path, samples[p99_idx])
        })
        .collect();
    top_slow_paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top_slow_paths.truncate(5);

    LogSummary {
        total,
        error_rate,
        status_counts,
        top_errors,
        top_slow_paths,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{ApacheRecord, LogRecord};

    /// Build a minimal ApacheRecord with only the status field set meaningfully.
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

    /// Build an ApacheRecord with a specific path, for top_errors testing.
    fn apache_record_path(status: u16, path: &str) -> LogRecord {
        LogRecord::Apache(ApacheRecord {
            ip: "127.0.0.1".to_string(),
            ident: "-".to_string(),
            auth: "-".to_string(),
            timestamp: "01/Jan/2026:00:00:00 +0000".to_string(),
            method: "GET".to_string(),
            path: path.to_string(),
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
        let summary = aggregate(vec![]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.error_rate, 0.0);
    }

    #[test]
    fn top_errors_groups_by_path() {
        let records = vec![
            apache_record_path(500, "/api/search"),
            apache_record_path(500, "/api/search"),
            apache_record_path(404, "/missing"),
            apache_record_path(200, "/ok"),
        ];
        let summary = aggregate(records);

        // /api/search appears twice (both are errors), /missing once.
        assert_eq!(summary.top_errors[0].0, "/api/search");
        assert_eq!(summary.top_errors[0].1, 2);
        assert_eq!(summary.top_errors[1].0, "/missing");
        assert_eq!(summary.top_errors[1].1, 1);

        // 200 responses must not appear in top_errors.
        assert!(!summary.top_errors.iter().any(|(p, _)| p == "/ok"));
    }

    #[test]
    fn apache_logs_have_no_slow_paths() {
        // Apache Combined Log Format has no latency field.
        let records = vec![apache_record(200), apache_record(500)];
        let summary = aggregate(records);
        assert!(summary.top_slow_paths.is_empty());
    }
}
