# log-lens

> Ingest any log file. Get a plain-English diagnosis. Ask follow-up questions.

log-lens is a Rust-native log analysis tool with an LLM reasoning layer. It parses log files of any format, computes aggregate statistics, and uses the Anthropic API to explain what's wrong — with a persistent Q&A interface for follow-up.

---

## Why this exists

Two primary problem definitions:
1. Engineering teams often consume time undertaking Bug Triage sessions to identify, prioritise and even fix bugs. This is not a glamorous part of engineering, and even armed with excellent ingestion platforms like Sentry and Bugsnag, there is still a lot of manual thought-work for engineering teams
2. Small non-tech companies often do not have in-house technical resource, but do have web presences and applications. Smarter log digesting could help them prioritise and communicate with their casual tech contractors.

This tool aims to act as an automated bug triage engineer, locating impacts, providing risk analysis and plain English explanations on what is most important to resolve and why.

Most log analysis tools require you to already know what you're looking for. log-lens inverts that: give it a log file you've never seen before, get a structured diagnosis, then interrogate it conversationally.

It handles format heterogeneity via AI-inferred parsing — meaning it works on log formats it has never seen, without user-supplied regex. The schema is cached after the first inference, so the API is called once per novel format, not once per run.

---

## Architecture

The stack is intentionally split by concern:

| Layer | Technology | Rationale |
|---|---|---|
| Parsing & aggregation | Rust | Zero-copy buffer slicing, typed domain model, safe parallel ingestion via Tokio |
| AI reasoning | `AnalysisEngine` trait → `AnthropicEngine` | Dependency-inverted abstraction; AI provider is a seam, not a dependency |
| Web interface | Axum + Chart.js | Single Rust runtime, no second-language deployment story |

**The key architectural decision** is the `AnalysisEngine` trait. Nothing outside `src/ai/` depends on `AnthropicEngine` directly. Swapping to a Python microservice, a different LLM, or a local model is a drop-in replacement — no changes outside `src/ai/`.

```rust
pub trait AnalysisEngine: Send + Sync {
    async fn analyse(&self, summary: &LogSummary) -> Result<AnalysisResult>;
    async fn chat(&self, history: &[Message], question: &str) -> Result<String>;
}
```

The LLM never receives raw log lines. It receives a compact `LogSummary` struct produced by the Rust aggregation pipeline. This keeps token usage minimal, forces meaningful computation to happen in Rust, and means the system works without real production data.

---

## Log ingestion: tiered parsing

```
Tier 1 — explicit --format flag     → built-in parser, no API call
Tier 2 — auto-detect (first 5 lines) → built-in parser heuristics, no API call
Tier 3 — no match                   → LLM infers schema from 20 sample lines, cached locally
```

Built-in formats: Apache Combined, Nginx access, JSON structured, Syslog.

---

## Project structure

```
src/
├── main.rs              # CLI entrypoint (clap), mode routing
├── parser/
│   ├── mod.rs           # Parser trait + LogRecord enum
│   ├── apache.rs
│   ├── nginx.rs
│   ├── json.rs
│   └── ai_infer.rs      # AI-inferred format fallback + schema cache
├── aggregator.rs        # Pure stats: error rates, status codes, latency percentiles
├── summary.rs           # LogSummary → compact JSON for AI
├── ai/
│   ├── mod.rs           # AnalysisEngine trait — the abstraction boundary
│   ├── anthropic.rs     # Concrete impl, reads ANTHROPIC_API_KEY from env
│   └── mock.rs          # Canned responses for tests and offline dev
└── server.rs            # Axum: REST + SSE endpoints
frontend/
└── index.html           # Single-file UI, no build step
samples/
└── apache_sample.log    # Synthetic data only
```

---

## Getting started

**Prerequisites**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
export ANTHROPIC_API_KEY=sk-ant-...
```

**Build**

```bash
cargo new log-lens && cd log-lens
# Add dependencies per Cargo.toml
cargo build
```

**Usage**

```bash
# Auto-detect format, print plain-English report
cargo run -- --file samples/apache_sample.log

# Force format
cargo run -- --file access.log --format apache

# Interactive Q&A in terminal
cargo run -- --file access.log --interactive

# Web interface at localhost:3000
cargo run -- --file access.log --serve
```

---

## Dependencies

| Purpose | Crate |
|---|---|
| Async runtime | `tokio` (full features) |
| Web server | `axum 0.7` |
| HTTP client | `reqwest 0.12` |
| Serialisation | `serde` + `serde_json` |
| Regex | `regex` |
| CLI | `clap 4` |
| Date/time | `chrono` |
| Error handling | `anyhow` |
| Middleware | `tower-http` (fs, cors) |

---

## Future evolution

The current architecture doesn't foreclose:

- **Streaming ingestion** — async `BufReader` + `lines()` from the start; no full-file-load assumption
- **High volume** — zero-copy buffer slicing; no `String` allocation per parsed field
- **Multiple simultaneous sources** — `Parser` trait is object-safe; `Aggregator` accepts streams
- **Python AI microservice** — implement `AnalysisEngine` as an HTTP client; everything else unchanged
- **Different LLM providers** — same seam; new provider is a new `impl`, not a refactor
- **Persistent storage** — all summary and history structs derive `Serialize` from day one

---

## Intentional Design constraints 

- Raw log lines are never sent to the LLM
- `AnthropicEngine` is not visible outside `src/ai/`
- `samples/` contains only synthetic data
- The web server is Rust/Axum — not FastAPI — to maintain a single runtime

---

## License

MIT
