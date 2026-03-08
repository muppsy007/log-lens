use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};
use std::str::FromStr;

use crate::aggregator::LogSummary;
use crate::ai::AnalysisResult;

/// Lightweight row returned by `ResultStore::list` — no heavy JSON columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResult {
    pub id: i64,
    pub filename: String,
    pub analysed_at: DateTime<Utc>,
}

/// Full row returned by `ResultStore::get`, with deserialized summary and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResultDetail {
    pub id: i64,
    pub filename: String,
    pub analysed_at: DateTime<Utc>,
    /// Deserialized from the JSON `summary` column.
    pub summary: LogSummary,
    /// Deserialized from the JSON `analysis` column.
    pub analysis: AnalysisResult,
}

/// Wraps a SQLite connection pool and owns the schema lifecycle.
///
/// `SqlitePool` is `Clone` (it is internally `Arc`-backed), so `ResultStore`
/// can be cheaply cloned and stored in Axum shared state without an extra `Arc`.
#[derive(Clone)]
pub struct ResultStore {
    pool: SqlitePool,
}

impl ResultStore {
    /// Opens (or creates) the database at `db_path` and runs the schema migration.
    ///
    /// `db_path` may be:
    /// - A bare file path: `./log_lens.db`   — the `sqlite:` prefix is added automatically.
    /// - A full SQLite URL: `sqlite:./log_lens.db`
    /// - The in-memory sentinel: `sqlite::memory:` (used in tests).
    pub async fn new(db_path: &str) -> Result<Self> {
        let url = if db_path.starts_with("sqlite:") {
            db_path.to_string()
        } else {
            format!("sqlite:{db_path}")
        };

        let opts = SqliteConnectOptions::from_str(&url)?.create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;

        // Idempotent schema — safe to run on every startup.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS analysis_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                analysed_at TEXT NOT NULL,
                summary     TEXT NOT NULL,
                analysis    TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Persists a completed analysis and returns the new row's `id`.
    pub async fn save(
        &self,
        filename: &str,
        summary: &LogSummary,
        analysis: &AnalysisResult,
    ) -> Result<i64> {
        // Store the timestamp as an explicit RFC3339 string so the format is
        // deterministic regardless of sqlx version or platform.
        let analysed_at = Utc::now().to_rfc3339();
        let summary_json = serde_json::to_string(summary)?;
        let analysis_json = serde_json::to_string(analysis)?;

        let res = sqlx::query(
            "INSERT INTO analysis_results (filename, analysed_at, summary, analysis)
             VALUES (?, ?, ?, ?)",
        )
        .bind(filename)
        .bind(&analysed_at)
        .bind(&summary_json)
        .bind(&analysis_json)
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    /// Returns all stored results, newest first, without the heavy JSON columns.
    pub async fn list(&self) -> Result<Vec<StoredResult>> {
        let rows = sqlx::query(
            "SELECT id, filename, analysed_at FROM analysis_results ORDER BY id DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|row| {
                let id: i64 = row.get("id");
                let filename: String = row.get("filename");
                let analysed_at = parse_timestamp(row.get("analysed_at"))?;
                Ok(StoredResult { id, filename, analysed_at })
            })
            .collect()
    }

    /// Returns the full stored result for `id`, or `None` if it does not exist.
    pub async fn get(&self, id: i64) -> Result<Option<StoredResultDetail>> {
        let maybe_row = sqlx::query(
            "SELECT id, filename, analysed_at, summary, analysis
             FROM analysis_results WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = maybe_row else {
            return Ok(None);
        };

        let id: i64 = row.get("id");
        let filename: String = row.get("filename");
        let analysed_at = parse_timestamp(row.get("analysed_at"))?;
        let summary_json: String = row.get("summary");
        let analysis_json: String = row.get("analysis");

        let summary: LogSummary = serde_json::from_str(&summary_json)
            .map_err(|e| anyhow!("Failed to deserialise summary: {e}"))?;
        let analysis: AnalysisResult = serde_json::from_str(&analysis_json)
            .map_err(|e| anyhow!("Failed to deserialise analysis: {e}"))?;

        Ok(Some(StoredResultDetail { id, filename, analysed_at, summary, analysis }))
    }
}

/// Parses an RFC3339 timestamp string stored in the database.
fn parse_timestamp(s: String) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| anyhow!("Invalid timestamp {s:?}: {e}"))
}
