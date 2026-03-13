mod aggregator;
mod ai;
mod parser;
mod pipeline;
pub mod server;
mod store;

use anyhow::Result;
// The derive macro is re-exported as `clap::Parser`; aliasing it avoids a
// name collision with our own `parser::Parser` trait in the same scope.
use clap::Parser as ClapParser;
use colored::Colorize;

use ai::{anthropic::AnthropicEngine, AnalysisEngine};

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
    let (out, skipped) = pipeline::parse_and_aggregate(file_path, |stage, msg| {
        match stage {
            "format_unknown" => println!("{}", msg.yellow()),
            "format_known" | "format_cached" => println!("{}", msg.dimmed()),
            _ => {}
        }
    })
    .await?;

    if skipped > 0 {
        eprintln!("{}", format!("Warning: skipped {skipped} malformed lines").yellow());
    }

    if out.summary.total == 0 {
        eprintln!("{}", "No valid log records found — check the file format.".red());
        std::process::exit(1);
    }

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
