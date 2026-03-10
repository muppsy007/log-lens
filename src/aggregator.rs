use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

use crate::parser::LogRecord;

/// Sample log lines associated with a particular error pattern.
/// Used to attach representative evidence to triage issues without ever
/// sending raw lines to the LLM (the join happens post-analysis).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ErrorEvidence {
    /// The error path (or message prefix) used as the grouping key.
    pub pattern: String,
    /// Total number of error occurrences for this pattern.
    pub count: u32,
    /// Up to 10 raw log lines that matched this pattern.
    pub sample_lines: Vec<String>,
}

/// Combined output of the aggregation pipeline.
///
/// `summary` is the data that gets serialised and sent to the LLM.
/// `evidence` is kept separate — it contains raw log lines that must
/// NEVER be included in any LLM payload.
#[derive(Debug, Clone)]
pub struct AggregatorOutput {
    pub summary: LogSummary,
    pub evidence: Vec<ErrorEvidence>,
}

/// Structured detail about a recurring error, extracted from parsed log records.
/// Sent to the LLM so it can reference specific messages, files, and line numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// The error message text (deduplication key).
    pub message: String,
    /// Source file reported by the logger, if present.
    pub file: Option<String>,
    /// Source line number reported by the logger, if present.
    pub line: Option<u32>,
    /// Severity level string: ERROR, WARNING, CRITICAL, etc.
    pub level: String,
    /// Number of times this exact message appeared in the batch.
    pub count: u32,
}

/// An IP address flagged as suspicious by one or more heuristics.
///
/// Serialised into `LogSummary` so the AI layer can reference specific IPs
/// in triage output without any extra API calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousIp {
    pub ip: String,
    /// Human-readable description of why this IP was flagged.
    /// Multiple reasons are joined with ", " when an IP triggers more than one heuristic.
    pub reason: String,
    /// Total requests from this IP across the log batch.
    pub request_count: u32,
}

/// The statistical summary produced from a batch of parsed log records.
/// Kept as a plain data struct (not an iterator/stream) so it can be serialised
/// to JSON, stored, or handed to the AI layer without further transformation.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSummary {
    /// Total number of records in the batch.
    pub total: u64,
    /// Fraction of records that returned a 5xx (server error) status code.
    /// Defined as 5xx-only — client errors (4xx) are tracked in status_counts
    /// but excluded from error_rate so operational alerting stays focused on
    /// server-side failures rather than normal 404/401 traffic.
    pub error_rate: f64,
    /// Count of records grouped by HTTP status code.
    #[serde_as(as = "HashMap<DisplayFromStr, _>")]
    pub status_counts: HashMap<u16, u64>,
    /// Top 20 errors by frequency, with structured metadata where available.
    /// For Apache records this is keyed by request path; for inferred records
    /// the message, file, and line fields are extracted from the parsed fields.
    #[serde(default)]
    pub top_errors: Vec<ErrorDetail>,
    /// Top 5 paths by p99 latency (path, p99_latency_ms).
    /// Empty when the parsed records contain no latency data.
    #[serde(default)]
    pub top_slow_paths: Vec<(String, f64)>,
    /// IPs flagged by security heuristics (traversal attempts, auth probing, etc.).
    /// Included in the summary JSON sent to the LLM so it can name specific IPs.
    #[serde(default)]
    pub suspicious_ips: Vec<SuspiciousIp>,
}

/// Returns true if `path` contains patterns associated with path traversal attacks.
fn is_traversal_path(path: &str) -> bool {
    path.contains("../") || path.contains("..\\") || path.contains("etc/passwd")
}

/// Aggregates a batch of parsed log records into a `LogSummary` plus raw-line evidence.
///
/// Accepts ownership of the `Vec` so callers make the move/clone decision
/// explicitly; we never need to read the records again after aggregation.
pub fn aggregate(records: Vec<LogRecord>) -> AggregatorOutput {
    let total = records.len() as u64;

    let mut status_counts: HashMap<u16, u64> = HashMap::new();
    // message/path → ErrorDetail (deduplication key is the message string)
    let mut error_details: HashMap<String, ErrorDetail> = HashMap::new();
    // message/path → up to 10 raw log lines for that error pattern
    let mut error_path_samples: HashMap<String, Vec<String>> = HashMap::new();
    // path → collected latency samples for p99 computation
    let mut latency_samples: HashMap<String, Vec<f64>> = HashMap::new();

    // ── Per-IP suspicious activity tracking ──────────────────────────────
    // Total requests seen from each IP.
    let mut ip_total: HashMap<String, u32> = HashMap::new();
    // Number of 401 responses received per IP.
    let mut ip_auth_failures: HashMap<String, u32> = HashMap::new();
    // IPs that made at least one request with a traversal path pattern.
    let mut ip_traversal: HashSet<String> = HashSet::new();
    // Up to 10 raw log lines per IP (for suspicious IP evidence).
    let mut ip_samples: HashMap<String, Vec<String>> = HashMap::new();

    for record in &records {
        match record {
            LogRecord::Apache(r) => {
                // Only count genuine HTTP status codes (100–599).
                if (100..600).contains(&r.status) {
                    *status_counts.entry(r.status).or_insert(0) += 1;
                }

                if r.status >= 400 {
                    let level = if r.status >= 500 { "ERROR" } else { "WARNING" }.to_string();
                    let entry = error_details.entry(r.path.clone()).or_insert_with(|| ErrorDetail {
                        message: r.path.clone(),
                        file: None,
                        line: None,
                        level,
                        count: 0,
                    });
                    entry.count += 1;
                    let samples = error_path_samples.entry(r.path.clone()).or_default();
                    if samples.len() < 10 {
                        samples.push(r.raw.clone());
                    }
                }

                // Apache Combined Log Format carries no latency field;
                // top_slow_paths will be empty for pure Apache logs.

                // Suspicious-IP heuristics — Apache records carry a typed `ip` field.
                *ip_total.entry(r.ip.clone()).or_insert(0) += 1;
                let ip_s = ip_samples.entry(r.ip.clone()).or_default();
                if ip_s.len() < 10 {
                    ip_s.push(r.raw.clone());
                }

                if r.status == 401 {
                    *ip_auth_failures.entry(r.ip.clone()).or_insert(0) += 1;
                }

                if is_traversal_path(&r.path) {
                    ip_traversal.insert(r.ip.clone());
                }
            }
            LogRecord::Inferred(r) => {
                // "code" is intentionally excluded — PHP error constants use a
                // `code` field (e.g. 8192 = E_DEPRECATED, 512 = E_USER_ERROR)
                // that must not be mistaken for HTTP status codes.
                let status = ["status", "status_code", "http_status"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                    .and_then(|v| v.parse::<u16>().ok());

                // Only count genuine HTTP status codes (100–599).
                if let Some(s) = status.filter(|&s| (100..600).contains(&s)) {
                    *status_counts.entry(s).or_insert(0) += 1;

                    if s >= 400 {
                        let path_key = ["path", "url", "uri", "request_path"]
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
                        let level = if s >= 500 { "ERROR" } else { "WARNING" }.to_string();
                        let entry = error_details.entry(path_key.clone()).or_insert_with(|| ErrorDetail {
                            message: path_key.clone(),
                            file: None,
                            line: None,
                            level,
                            count: 0,
                        });
                        entry.count += 1;
                        let samples = error_path_samples.entry(path_key).or_default();
                        if samples.len() < 10 {
                            samples.push(r.raw.clone());
                        }
                    }
                }

                // For application logs (no HTTP status), extract message/file/line/level.
                let level_val = ["level", "severity", "log_level", "loglevel", "priority"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                    .cloned();

                if status.is_none() || !(100..600).contains(&status.unwrap_or(0)) {
                    if let Some(ref level) = level_val {
                        let level_upper = level.to_uppercase();
                        let is_error_level = ["ERROR", "CRITICAL", "ALERT", "EMERGENCY", "WARNING", "WARN"]
                            .iter()
                            .any(|&l| level_upper.contains(l));
                        if is_error_level {
                            let message = ["message", "msg", "error", "exception", "text"]
                                .iter()
                                .find_map(|&k| r.fields.get(k))
                                .cloned()
                                .unwrap_or_else(|| r.raw.chars().take(80).collect());
                            let key = message.chars().take(120).collect::<String>();
                            let file = ["file", "filename", "source_file", "src_file"]
                                .iter()
                                .find_map(|&k| r.fields.get(k))
                                .cloned();
                            let line = ["line", "lineno", "line_number", "src_line"]
                                .iter()
                                .find_map(|&k| r.fields.get(k))
                                .and_then(|v| v.parse::<u32>().ok());
                            let entry = error_details.entry(key.clone()).or_insert_with(|| ErrorDetail {
                                message: key.clone(),
                                file,
                                line,
                                level: level_upper,
                                count: 0,
                            });
                            entry.count += 1;
                            let samples = error_path_samples.entry(key).or_default();
                            if samples.len() < 10 {
                                samples.push(r.raw.clone());
                            }
                        }
                    }
                }

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

                // Suspicious-IP heuristics — Inferred records store IP in free-form fields.
                if let Some(ip) = ["ip", "client_ip", "remote_addr", "client", "host"]
                    .iter()
                    .find_map(|&k| r.fields.get(k))
                {
                    *ip_total.entry(ip.clone()).or_insert(0) += 1;
                    let ip_s = ip_samples.entry(ip.clone()).or_default();
                    if ip_s.len() < 10 {
                        ip_s.push(r.raw.clone());
                    }

                    if status == Some(401) {
                        *ip_auth_failures.entry(ip.clone()).or_insert(0) += 1;
                    }

                    if let Some(path) = ["path", "url", "uri", "request_path"]
                        .iter()
                        .find_map(|&k| r.fields.get(k))
                    {
                        if is_traversal_path(path) {
                            ip_traversal.insert(ip.clone());
                        }
                    }
                }
            }
        }
    }

    // ── Derive suspicious IPs from per-IP stats ───────────────────────────
    let mut suspicious_ips: Vec<SuspiciousIp> = ip_total
        .iter()
        .filter_map(|(ip, &count)| {
            let mut reasons: Vec<&str> = Vec::new();

            if ip_traversal.contains(ip) {
                reasons.push("path traversal attempt");
            }
            if ip_auth_failures.get(ip).copied().unwrap_or(0) >= 3 {
                reasons.push("repeated auth failures");
            }

            if reasons.is_empty() {
                None
            } else {
                Some(SuspiciousIp {
                    ip: ip.clone(),
                    reason: reasons.join(", "),
                    request_count: count,
                })
            }
        })
        .collect();

    // Stable sort so the LLM receives a deterministic list.
    suspicious_ips.sort_by(|a, b| a.ip.cmp(&b.ip));

    // ── Summary statistics ────────────────────────────────────────────────

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

    let mut top_errors: Vec<ErrorDetail> = error_details.into_values().collect();
    top_errors.sort_by(|a, b| b.count.cmp(&a.count));
    top_errors.truncate(20);

    // Build evidence from the top error entries (in the same order as top_errors).
    let mut evidence: Vec<ErrorEvidence> = top_errors
        .iter()
        .map(|detail| {
            let sample_lines = error_path_samples.remove(&detail.message).unwrap_or_default();
            ErrorEvidence { pattern: detail.message.clone(), count: detail.count, sample_lines }
        })
        .collect();

    // Also add evidence for each suspicious IP.
    for suspicious in &suspicious_ips {
        let sample_lines = ip_samples.remove(&suspicious.ip).unwrap_or_default();
        evidence.push(ErrorEvidence {
            pattern: suspicious.ip.clone(),
            count: suspicious.request_count,
            sample_lines,
        });
    }

    let mut top_slow_paths: Vec<(String, f64)> = latency_samples
        .into_iter()
        .map(|(path, mut samples)| {
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = (samples.len() * 99 / 100).min(samples.len() - 1);
            (path, samples[p99_idx])
        })
        .collect();
    top_slow_paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top_slow_paths.truncate(5);

    let summary = LogSummary {
        total,
        error_rate,
        status_counts,
        top_errors,
        top_slow_paths,
        suspicious_ips,
    };

    AggregatorOutput { summary, evidence }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{ApacheRecord, LogRecord};

    fn apache_record(status: u16) -> LogRecord {
        apache_record_ip_path(status, "127.0.0.1", "/")
    }

    fn apache_record_path(status: u16, path: &str) -> LogRecord {
        apache_record_ip_path(status, "127.0.0.1", path)
    }

    fn apache_record_ip_path(status: u16, ip: &str, path: &str) -> LogRecord {
        let raw = format!(
            r#"{ip} - - [01/Jan/2026:00:00:00 +0000] "GET {path} HTTP/1.1" {status} 512 "-" "test-agent""#
        );
        LogRecord::Apache(ApacheRecord {
            ip: ip.to_string(),
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
            raw,
        })
    }

    #[test]
    fn error_rate_counts_only_5xx() {
        let records = vec![
            apache_record(200),
            apache_record(200),
            apache_record(404),
            apache_record(500),
        ];
        let out = aggregate(records);
        assert_eq!(out.summary.total, 4);
        assert!((out.summary.error_rate - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn status_counts_are_correct() {
        let records = vec![
            apache_record(200),
            apache_record(200),
            apache_record(404),
            apache_record(500),
        ];
        let out = aggregate(records);
        assert_eq!(*out.summary.status_counts.get(&200).unwrap(), 2);
        assert_eq!(*out.summary.status_counts.get(&404).unwrap(), 1);
        assert_eq!(*out.summary.status_counts.get(&500).unwrap(), 1);
    }

    #[test]
    fn empty_input_returns_zero_error_rate() {
        let out = aggregate(vec![]);
        assert_eq!(out.summary.total, 0);
        assert_eq!(out.summary.error_rate, 0.0);
    }

    #[test]
    fn top_errors_groups_by_path() {
        let records = vec![
            apache_record_path(500, "/api/search"),
            apache_record_path(500, "/api/search"),
            apache_record_path(404, "/missing"),
            apache_record_path(200, "/ok"),
        ];
        let out = aggregate(records);
        assert_eq!(out.summary.top_errors[0].message, "/api/search");
        assert_eq!(out.summary.top_errors[0].count, 2);
        assert_eq!(out.summary.top_errors[1].message, "/missing");
        assert_eq!(out.summary.top_errors[1].count, 1);
        assert!(!out.summary.top_errors.iter().any(|e| e.message == "/ok"));
    }

    #[test]
    fn status_counts_excludes_non_http_codes() {
        use crate::parser::{InferredRecord, LogRecord};
        // PHP error constants stored in a `code` field (8192 = E_DEPRECATED,
        // 512 = E_USER_ERROR) must not appear in status_counts even when `code`
        // is in the range 100–599.
        let mut fields = HashMap::new();
        fields.insert("code".to_string(), "512".to_string());
        fields.insert("message".to_string(), "PHP user error".to_string());
        let record_512 = LogRecord::Inferred(InferredRecord {
            fields: fields.clone(),
            raw: "php user error line".to_string(),
        });
        let mut fields2 = HashMap::new();
        fields2.insert("code".to_string(), "8192".to_string());
        fields2.insert("message".to_string(), "PHP deprecated".to_string());
        let record_8192 = LogRecord::Inferred(InferredRecord {
            fields: fields2,
            raw: "php deprecated line".to_string(),
        });
        let out = aggregate(vec![record_512, record_8192]);
        assert!(!out.summary.status_counts.contains_key(&512u16), "512 must not be a status code");
        assert!(!out.summary.status_counts.contains_key(&8192u16), "8192 must not be a status code");
    }

    #[test]
    fn apache_logs_have_no_slow_paths() {
        let records = vec![apache_record(200), apache_record(500)];
        let out = aggregate(records);
        assert!(out.summary.top_slow_paths.is_empty());
    }

    #[test]
    fn traversal_path_flags_ip() {
        let records = vec![
            apache_record_ip_path(400, "10.0.0.99", "/../../../../etc/passwd"),
            apache_record_ip_path(200, "192.168.1.10", "/index.html"),
        ];
        let out = aggregate(records);
        assert_eq!(out.summary.suspicious_ips.len(), 1);
        assert_eq!(out.summary.suspicious_ips[0].ip, "10.0.0.99");
        assert!(out.summary.suspicious_ips[0].reason.contains("path traversal"));
        assert_eq!(out.summary.suspicious_ips[0].request_count, 1);
    }

    #[test]
    fn repeated_401s_flags_ip() {
        let records = vec![
            apache_record_ip_path(401, "10.0.0.5", "/api/auth"),
            apache_record_ip_path(401, "10.0.0.5", "/api/auth"),
            apache_record_ip_path(401, "10.0.0.5", "/admin/login"),
            apache_record_ip_path(200, "192.168.1.22", "/index.html"),
        ];
        let out = aggregate(records);
        assert_eq!(out.summary.suspicious_ips.len(), 1);
        assert_eq!(out.summary.suspicious_ips[0].ip, "10.0.0.5");
        assert!(out.summary.suspicious_ips[0].reason.contains("repeated auth failures"));
        assert_eq!(out.summary.suspicious_ips[0].request_count, 3);
    }

    #[test]
    fn two_401s_does_not_flag_ip() {
        // Threshold is 3 — two 401s should not be flagged.
        let records = vec![
            apache_record_ip_path(401, "10.0.0.5", "/api/auth"),
            apache_record_ip_path(401, "10.0.0.5", "/api/auth"),
            apache_record_ip_path(200, "10.0.0.5", "/index.html"),
        ];
        let out = aggregate(records);
        assert!(out.summary.suspicious_ips.is_empty());
    }

    #[test]
    fn clean_traffic_has_no_suspicious_ips() {
        let records = vec![
            apache_record_ip_path(200, "192.168.1.10", "/index.html"),
            apache_record_ip_path(200, "192.168.1.22", "/api/data"),
            apache_record_ip_path(404, "192.168.1.31", "/missing"),
        ];
        let out = aggregate(records);
        assert!(out.summary.suspicious_ips.is_empty());
    }
}
