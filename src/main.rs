mod aggregator;
mod ai;
mod parser;
mod summary;

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
// The derive macro is re-exported as `clap::Parser`; aliasing it here avoids
// a name collision with our own `parser::Parser` trait in the same scope.
use clap::Parser as ClapParser;
use colored::Colorize;

use aggregator::aggregate;
use ai::{anthropic::AnthropicEngine, AnalysisEngine};
use parser::ai_infer::AiInferredParser;
use parser::apache::ApacheParser;
// Import our Parser trait so `.parse()` is in scope on both parser types.
use parser::Parser;
use summary::to_json;

/// CLI configuration. `clap` derives argument parsing from this struct,
/// turning field names into `--flag` names automatically.
#[derive(ClapParser)]
#[command(name = "log-lens", about = "Analyse log files using an LLM")]
struct Cli {
    /// Path to the log file to analyse.
    #[arg(long)]
    file: String,
}

// `#[tokio::main]` rewrites `main` into a synchronous entry point that
// bootstraps the Tokio runtime and then drives the async body to completion.
// Without this macro, `.await` would not be usable in `main`.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // -----------------------------------------------------------------------
    // 1. Read file into memory
    // -----------------------------------------------------------------------

    let file = File::open(&cli.file)
        // `.with_context()` attaches the file path to any IO error so the
        // user sees "Could not open file: foo.log" instead of a bare OS error.
        .with_context(|| format!("Could not open file: {}", cli.file))?;

    // Collect all lines upfront so we can sample them for format detection
    // before choosing a parser, without reading the file twice.
    // `BufReader` buffers the read to avoid per-byte OS syscalls.
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        // `collect::<io::Result<Vec<_>>>()` short-circuits on the first IO error.
        .collect::<std::io::Result<Vec<_>>>()?;

    // -----------------------------------------------------------------------
    // 2. Format detection (Tier 2 → Tier 3)
    // -----------------------------------------------------------------------

    // Sample up to 20 non-empty lines for detection.
    let sample: Vec<&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .take(20)
        .map(String::as_str)
        .collect();

    if sample.is_empty() {
        eprintln!("{}", "No non-empty lines found in file.".red());
        std::process::exit(1);
    }

    // Tier 2: heuristic check — try Apache parser on the first 5 sample lines.
    // Five lines is enough to rule out coincidental partial matches on a header.
    let apache_parser = ApacheParser::new()?;
    let tier2_sample = &sample[..sample.len().min(5)];
    let apache_hits = tier2_sample
        .iter()
        .filter(|l| apache_parser.parse(l).is_ok())
        .count();

    // Use Apache if the majority of the sample matches; otherwise fall through
    // to Tier 3. The `* 2 >= len` trick avoids floating-point division.
    let parser: Box<dyn Parser> = if apache_hits * 2 >= tier2_sample.len() {
        println!("{}", "Detected format: Apache Combined Log".dimmed());
        Box::new(apache_parser)
    } else {
        // Tier 3: send up to 20 sample lines to the LLM to infer a regex.
        // The result is cached to disk, so the API is called only once per
        // novel format regardless of how many times the file is processed.
        println!("{}", "Format not recognised — asking LLM to infer schema…".yellow());
        Box::new(AiInferredParser::new(&sample).await?)
    };

    // -----------------------------------------------------------------------
    // 3. Parse all lines with the selected parser
    // -----------------------------------------------------------------------

    let mut records = Vec::new();
    let mut skipped: usize = 0;

    for line in &lines {
        if line.trim().is_empty() {
            continue;
        }
        match parser.parse(line) {
            Ok(record) => records.push(record),
            // Skip malformed lines silently and count them for the summary.
            // Failing the whole run on one bad line would be too brittle for
            // real log files, which often contain partial or mixed-format lines.
            Err(_) => skipped += 1,
        }
    }

    if skipped > 0 {
        eprintln!("{}", format!("Warning: skipped {skipped} malformed lines").yellow());
    }

    if records.is_empty() {
        eprintln!("{}", "No valid log records found — check the file format.".red());
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // 4. Aggregate
    // -----------------------------------------------------------------------

    let summary = aggregate(records);
    let summary_json = to_json(&summary)?;

    println!("{}", format!("Analysing {} log records…", summary.total).bold());
    println!("{}", format!("  Error rate : {:.1}%", summary.error_rate * 100.0).dimmed());
    println!("{}", format!("  Summary    : {summary_json}").dimmed());
    println!();

    // -----------------------------------------------------------------------
    // 5. AI analysis via the trait object
    // -----------------------------------------------------------------------

    // Constructing `AnthropicEngine` reads the API key; if absent, it returns
    // an `Err` which we catch here to print a human-readable message and exit
    // rather than letting anyhow print a raw error chain.
    let engine: Box<dyn AnalysisEngine> = match AnthropicEngine::new() {
        Ok(e) => Box::new(e),
        Err(_) => {
            eprintln!(
                "{}",
                "Error: ANTHROPIC_API_KEY is not set.\n\
                 Export it and try again:\n\
                 \n  export ANTHROPIC_API_KEY=sk-ant-..."
                    .red()
            );
            std::process::exit(1);
        }
    };

    // Call through the trait object — main.rs is intentionally ignorant of
    // whether the engine is Anthropic, a mock, or a future Python microservice.
    let result = engine.analyse(&summary).await?;

    println!("{}", "Analysis".bold().underline());
    println!("{}", result.text);

    Ok(())
}
