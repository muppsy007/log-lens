# log-lens — Agent Context File

# For use with: Claude Code, Cursor, Copilot, or any AI coding assistant

## What This Project Is

`log-lens` is a CLI + web UI tool that ingests log files, aggregates statistics,
and uses an LLM to provide plain-English analysis and interactive Q&A.

Built in Rust. The AI layer is abstracted behind a trait so it can be swapped
or ported to a Python microservice without touching the parsing or aggregation layers.

## Why Rust

Rust was chosen deliberately for the parsing and aggregation pipeline:

- High-throughput line processing with zero-copy buffer slicing
- Typed, exhaustive domain model (LogRecord as enum variants per format)
- Async concurrency via Tokio for parallel file ingestion
- Architecture designed to scale to streaming ingestion and large log volumes

The web server (Axum) and API client (reqwest) are also Rust — keeping a single
runtime and clean deployment story.

## The AI Abstraction — Critical Design Decision

The AI layer is abstracted behind the `AnalysisEngine` trait:

```rust
trait AnalysisEngine {
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult>;
    async fn chat(&self, history: &[Message], question: &str) -> Result<String>;
}
```

The current implementation (`AnthropicEngine`) calls the Anthropic API directly via reqwest.

This boundary exists so the AI layer can be replaced with:

- A Python microservice (e.g. if LangChain or richer prompt orchestration is needed)
- A different LLM provider
- A local/self-hosted model
- A mock implementation for testing

**Never couple parsing, aggregation, or server logic directly to `AnthropicEngine`.
Always depend on the `AnalysisEngine` trait.**

## Log Ingestion Strategy

Tiered approach — do not collapse these tiers:

| Tier | Trigger | Behaviour |
|------|---------|-----------|
| 1 | Known format flag (`--format apache`) | Built-in parser used directly. No API call. |
| 2 | Auto-detect match | Sample first 5 lines, match against built-in parser heuristics. No API call. |
| 3 | No match (fallback) | Send 20 sample lines to LLM. Receive schema. Cache to disk. Apply to file. |

Built-in formats: Apache, Nginx, JSON, Syslog.

**Tier 2 must run before Tier 3.** The heuristic check is cheap; the API call is not.
The cached schema means the API is called once per novel format, not once per run.

## Project Structure

```
log-lens/
├── CLAUDE.md                  ← you are here
├── .cursor/rules/             ← Cursor agent rules
│   └── project.mdc
├── Cargo.toml
├── src/
│   ├── main.rs                # CLI entrypoint, clap config, mode routing
│   ├── parser/
│   │   ├── mod.rs             # Parser trait + format detection
│   │   ├── apache.rs          # Apache Combined Log Format
│   │   ├── nginx.rs           # Nginx access log format
│   │   ├── json.rs            # JSON structured logs
│   │   └── ai_infer.rs        # AI-inferred format fallback
│   ├── aggregator.rs          # Stats, error rates, spike detection, latency buckets
│   ├── summary.rs             # Builds compact JSON payload for AI (never raw lines)
│   ├── ai/
│   │   ├── mod.rs             # AnalysisEngine trait definition
│   │   ├── anthropic.rs       # AnthropicEngine — concrete implementation
│   │   └── mock.rs            # MockEngine — for tests
│   └── server.rs              # Axum web server, REST endpoints, SSE streaming
├── frontend/
│   └── index.html             # Single-file UI: charts + chat interface
└── samples/
    └── apache_sample.log      # Synthetic test data — never use real logs
```

## Key Constraints for Agents

- **Never send raw log lines to the LLM.** Always send the aggregated `LogSummary` struct.
- **Never hardcode the API key.** Always read from `ANTHROPIC_API_KEY` env var.
- **Never add a direct dependency on `AnthropicEngine` outside of `ai/` module.**
- **All log parsing must produce typed `LogRecord` structs** — no stringly-typed HashMaps.
- **Use `anyhow::Result` for error propagation** throughout — no `.unwrap()` in library code.
- **Tokio** is the async runtime — do not introduce a second runtime.
- The `AnalysisEngine` trait must remain the only interface the server and CLI use for AI.
- **Every parser module must have a `#[cfg(test)]` block** with at least one valid-input test and one malformed-input test using lines from `samples/`
- **No `.unwrap()` in library code** — tests are the one exception
- Never search for ANTHROPIC_API_KEY in the keychain or any secret store. If the key is missing, print an error and stop.

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...     # Required
LOG_LENS_PORT=3000               # Optional, default 3000
LOG_LENS_HOST=127.0.0.1          # Optional, default 127.0.0.1
```

## Running the Project

```bash
# Report mode
cargo run -- --file samples/apache_sample.log

# Interactive mode
cargo run -- --file samples/apache_sample.log --interactive

# Web UI mode
cargo run -- --file samples/apache_sample.log --serve
# Then open http://localhost:3000
```

## Cargo.toml Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
axum = "0.7"
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
regex = "1"
clap = { version = "4", features = ["derive"] }
chrono = "0.4"
anyhow = "1"
colored = "2"
tower-http = { version = "0.5", features = ["fs", "cors"] }
```

## Build Order (for agents starting from scratch)

Build one step at a time. Verify each step before proceeding to the next.

| # | File | Done When |
|---|------|-----------|
| 1 | `parser/mod.rs` | Parser trait compiles. `LogRecord` enum has at least an Apache variant. |
| 2 | `parser/apache.rs` | Single Apache log line parses to `LogRecord` without panic. |
| 3 | `aggregator.rs` | `Vec<LogRecord>` produces `LogSummary` with error rate and status code counts. |
| 4 | `summary.rs` | `LogSummary` serialises to JSON correctly. |
| 5 | `ai/mod.rs` | `AnalysisEngine` trait compiles. No implementation yet. |
| 6 | `ai/mock.rs` | `MockEngine` returns canned `AnalysisResult`. Tests pass. |
| 7 | `ai/anthropic.rs` | `AnthropicEngine` calls API, returns real analysis. Tested with sample file. |
| 8 | `main.rs` | `cargo run -- --file samples/apache-access.log` prints plain-English summary. |
| 9 | `parser/ai_infer.rs` | Unknown-format file parsed correctly. Schema cached to disk. |
| 10 | `server.rs` | `GET /api/summary` and `POST /api/chat` return valid JSON. |
| 11 | `frontend/index.html` | Browser shows charts and chat interface at `localhost:3000`. |

## Verification

Run after every step before moving on:
```bash
cargo test
cargo run -- --file samples/apache-access.log
```

Step 8 onwards: confirm no panic and output is human-readable.

## Future Evolution (inform architectural decisions now)

- **Streaming ingestion**: Tail live log files/sockets via tokio async streams
- **Large volume**: Zero-copy buffer slicing, batch aggregation windows
- **Multiple simultaneous sources**: Correlated event streams across services
- **Python AI microservice**: Drop-in replacement for AnthropicEngine via HTTP
- **Pluggable storage**: Persist summaries and chat history

These are not current requirements but architecture must not foreclose them.