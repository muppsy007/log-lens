mod aggregator;
mod ai;
mod parser;
pub mod server;
mod store;

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
// The derive macro is re-exported as `clap::Parser`; aliasing it avoids a
// name collision with our own `parser::Parser` trait in the same scope.
use clap::Parser as ClapParser;
use colored::Colorize;

use aggregator::aggregate;
use ai::{anthropic::AnthropicEngine, AnalysisEngine};
use parser::ai_infer::AiInferredParser;
use parser::apache::ApacheParser;
// Import our Parser trait so `.parse()` is in scope on both parser types.
use parser::Parser;

/// CLI configuration. Clap derives argument parsing from the struct fields.
#[derive(ClapParser)]
#[command(name = "log-lens", about = "Analyse log files using an LLM")]
struct Cli {
    /// Path to the log file to analyse (required in analysis mode).
    #[arg(long)]
    file: Option<String>,

    /// Start the Axum HTTP server instead of running a one-shot analysis.
    /// The log file is then supplied as a query parameter: ?file=path/to/log
    #[arg(long, default_value_t = false)]
    serve: bool,
}

// `#[tokio::main]` rewrites `main` into a sync entry point that bootstraps
// the Tokio runtime and drives the async body to completion.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.serve {
        return run_server().await;
    }

    // -----------------------------------------------------------------------
    // Analysis mode — --file is required when not in server mode
    // -----------------------------------------------------------------------
    let file_path = cli
        .file
        .ok_or_else(|| anyhow::anyhow!("--file is required in analysis mode (or use --serve)"))?;

    run_analysis(&file_path).await
}

// ---------------------------------------------------------------------------
// Server mode
// ---------------------------------------------------------------------------

async fn run_server() -> Result<()> {
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

    // Read host/port from environment variables with sensible defaults.
    // Using env vars (rather than CLI flags) matches the twelve-factor app
    // convention and makes the server easy to configure in containers.
    let host =
        std::env::var("LOG_LENS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("LOG_LENS_PORT")
        .ok()
        // `and_then` chains on Some; `unwrap_or` provides the default.
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    server::start_server(engine, &host, port).await
}

// ---------------------------------------------------------------------------
// Analysis mode (one-shot CLI)
// ---------------------------------------------------------------------------

async fn run_analysis(file_path: &str) -> Result<()> {
    // -----------------------------------------------------------------------
    // 1. Read file into memory
    // -----------------------------------------------------------------------

    let file = File::open(file_path)
        .with_context(|| format!("Could not open file: {file_path}"))?;

    // Collect all lines so we can sample for format detection before choosing
    // a parser without reading the file twice.
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?;

    // -----------------------------------------------------------------------
    // 2. Format detection (Tier 2 → Tier 3)
    // -----------------------------------------------------------------------

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

    let apache_parser = ApacheParser::new()?;
    let tier2_sample = &sample[..sample.len().min(5)];
    let apache_hits = tier2_sample
        .iter()
        .filter(|l| apache_parser.parse(l).is_ok())
        .count();

    let parser: Box<dyn Parser> = if apache_hits * 2 >= tier2_sample.len() {
        println!("{}", "Detected format: Apache Combined Log".dimmed());
        Box::new(apache_parser)
    } else {
        let (parser, cache_hit) = AiInferredParser::new(&sample).await?;
        if cache_hit {
            println!("{}", "Format not recognised — using cached schema…".dimmed());
        } else {
            println!(
                "{}",
                "Format not recognised — schema inferred and cached…".yellow()
            );
        }
        Box::new(parser)
    };

    // -----------------------------------------------------------------------
    // 3. Parse all lines
    // -----------------------------------------------------------------------

    let mut records = Vec::new();
    let mut skipped: usize = 0;

    for line in &lines {
        if line.trim().is_empty() {
            continue;
        }
        match parser.parse(line) {
            Ok(record) => records.push(record),
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

    let out = aggregate(records);
    let summary = out.summary;
    let summary_json = serde_json::to_string(&summary)?;

    println!("{}", format!("Analysing {} log records…", summary.total).bold());
    println!("{}", format!("  Error rate : {:.1}%", summary.error_rate * 100.0).dimmed());
    println!("{}", format!("  Summary    : {summary_json}").dimmed());
    println!();

    // -----------------------------------------------------------------------
    // 5. AI analysis
    // -----------------------------------------------------------------------

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

    // Call through the trait object — main.rs never references AnthropicEngine
    // methods directly; everything goes through the AnalysisEngine trait.
    let result = engine.analyse(&summary).await?;

    println!("{}", "Analysis".bold().underline());
    if result.issues.is_empty() {
        // Fallback: model returned prose instead of JSON.
        if let Some(raw) = &result.raw {
            println!("{raw}");
        }
    } else {
        for issue in &result.issues {
            println!(
                "[{}] {} — {}  → {}",
                issue.severity.to_uppercase(),
                issue.title,
                issue.explanation,
                issue.action
            );
        }
    }

    Ok(())
}
