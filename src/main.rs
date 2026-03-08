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
use parser::apache::ApacheParser;
// Import our Parser trait so `.parse()` is in scope on `ApacheParser`.
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
    // 1. Parse the log file
    // -----------------------------------------------------------------------

    let file = File::open(&cli.file)
        // `.with_context()` attaches the file path to any IO error so the
        // user sees "Could not open file: foo.log" instead of a bare OS error.
        .with_context(|| format!("Could not open file: {}", cli.file))?;

    // `BufReader` wraps the file to buffer reads, which is far more efficient
    // than reading one byte at a time through the OS syscall layer.
    let reader = BufReader::new(file);

    // Initialise once and reuse — the compiled regex lives in `ApacheParser`.
    let parser = ApacheParser::new()?;

    let mut records = Vec::new();
    let mut skipped: usize = 0;

    for line in reader.lines() {
        // Each `line` is `io::Result<String>`; the `?` propagates IO errors
        // (e.g. permission denied mid-read) while we handle parse errors below.
        let line = line?;

        if line.trim().is_empty() {
            continue;
        }

        match parser.parse(&line) {
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
    // 2. Aggregate
    // -----------------------------------------------------------------------

    let summary = aggregate(records);

    // Serialise to JSON for diagnostic printing; the `_` prefix signals that
    // the binding is intentionally unused in the happy path.
    let summary_json = to_json(&summary)?;

    println!("{}", format!("Analysing {} log records…", summary.total).bold());
    println!("{}", format!("  Error rate : {:.1}%", summary.error_rate * 100.0).dimmed());
    println!("{}", format!("  Summary    : {summary_json}").dimmed());
    println!();

    // -----------------------------------------------------------------------
    // 3. AI analysis via the trait object
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
