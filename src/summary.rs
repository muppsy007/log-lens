use anyhow::Result;

use crate::aggregator::LogSummary;

/// Serialises a `LogSummary` to a compact JSON string.
///
/// Accepts a shared reference (`&LogSummary`) rather than ownership so the
/// caller can continue using the summary after serialisation (e.g. to pass it
/// to the AI layer or store it).
///
/// Returns `anyhow::Result<String>` so serialisation errors propagate cleanly
/// via `?` in callers without a manual `map_err`.
pub fn to_json(summary: &LogSummary) -> Result<String> {
    // `serde_json::to_string` produces compact JSON (no whitespace).
    // The error type is `serde_json::Error`, which anyhow converts automatically
    // via its `From<std::error::Error>` blanket impl — no `.map_err` needed.
    let json = serde_json::to_string(summary)?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::aggregator::LogSummary;

    #[test]
    fn serialises_known_summary_to_json() {
        let mut status_counts = HashMap::new();
        status_counts.insert(200u16, 3u64);
        status_counts.insert(500u16, 1u64);

        let summary = LogSummary {
            total: 4,
            // 1 out of 4 records is a 5xx error
            error_rate: 0.25,
            status_counts,
            top_errors: vec![],
            top_slow_paths: vec![],
        };

        let json = to_json(&summary).expect("serialisation must succeed");

        // Parse back to a generic Value so the test is not sensitive to key
        // ordering (HashMap iteration order is non-deterministic).
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("output must be valid JSON");

        assert_eq!(value["total"], 4);
        // f64 round-trip: compare via as_f64() to avoid floating-point string format assumptions.
        assert!((value["error_rate"].as_f64().unwrap() - 0.25).abs() < f64::EPSILON);
        assert_eq!(value["status_counts"]["200"], 3);
        assert_eq!(value["status_counts"]["500"], 1);
    }
}
