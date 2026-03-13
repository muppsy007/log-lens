use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};

use crate::aggregator::{aggregate, AggregatorOutput};
use crate::parser::ai_infer::AiInferredParser;
use crate::parser::apache::ApacheParser;
use crate::parser::Parser;

/// Runs the log parsing and aggregation pipeline (Tiers 2 and 3).
///
/// Tier 1 (explicit `--format` flag bypassing detection) is not yet
/// implemented; when it is, the caller will select the parser before
/// invoking this function and pass it in directly.
///
/// The two detection tiers handled here are:
///   - Tier 2 — Apache Combined Log heuristic (no API call)
///   - Tier 3 — LLM-inferred schema fallback (API call, result cached to disk)
///
/// `on_progress` is called at each stage with `(stage, message)`. Pass a
/// no-op closure (`|_, _| {}`) when progress reporting is not needed.
///
/// Returns the aggregated output and the count of lines that failed to parse.
pub async fn parse_and_aggregate<F>(
    path: &str,
    mut on_progress: F,
) -> Result<(AggregatorOutput, usize)>
where
    F: FnMut(&str, &str) + Send,
{
    on_progress("reading", &format!("Reading log file: {path}"));

    let file = File::open(path).with_context(|| format!("Could not open file: {path}"))?;
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?;

    on_progress("detecting", "Sampling lines for format detection");

    let sample: Vec<&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .take(20)
        .map(String::as_str)
        .collect();

    if sample.is_empty() {
        return Err(anyhow::anyhow!("No non-empty lines found in {:?}", path));
    }

    let apache_parser = ApacheParser::new()?;
    let tier2 = &sample[..sample.len().min(5)];
    let hits = tier2.iter().filter(|l| apache_parser.parse(l).is_ok()).count();
    let use_apache = hits * 2 >= tier2.len();

    // Resolve the inferred parser (if needed) before the synchronous record
    // collection block below. `Box<dyn Parser>` is not `Send`, so it must not
    // be held across any `.await` point. We await here while holding only
    // concrete `Send` types, then box inside the sync block.
    let inferred_parser: Option<AiInferredParser> = if use_apache {
        on_progress("format_known", "Detected format: Apache Combined Log");
        None
    } else {
        on_progress("detecting", "Checking schema cache for unknown format");
        let (parser, cache_hit) = AiInferredParser::new(&sample).await?;
        if cache_hit {
            on_progress("format_cached", "Format not recognised — using cached schema\u{2026}");
        } else {
            on_progress("format_unknown", "Format not recognised — schema inferred and cached\u{2026}");
        }
        Some(parser)
    };

    let n = lines.iter().filter(|l| !l.trim().is_empty()).count();
    on_progress("parsing", &format!("Parsing {n} log records"));

    // Collect records synchronously — no `.await` inside this block so
    // `Box<dyn Parser>` does not need to be `Send`.
    let (records, skipped) = {
        let parser: Box<dyn Parser> = match inferred_parser {
            Some(p) => Box::new(p),
            None => Box::new(apache_parser),
        };
        let mut ok = Vec::new();
        let mut skipped: usize = 0;
        for line in lines.iter().filter(|l| !l.trim().is_empty()) {
            match parser.parse(line) {
                Ok(r) => ok.push(r),
                Err(_) => skipped += 1,
            }
        }
        (ok, skipped)
        // parser dropped here, before any further await
    };

    on_progress("aggregating", "Aggregating statistics");

    Ok((aggregate(records), skipped))
}
