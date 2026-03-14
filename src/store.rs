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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::aggregator::LogSummary;
    use crate::ai::{AnalysisResult, Issue, Severity};

    fn stub_summary() -> LogSummary {
        let mut status_counts = HashMap::new();
        status_counts.insert(200u16, 10u64);
        LogSummary {
            total: 10,
            error_rate: 0.0,
            status_counts,
            top_errors: vec![],
            top_slow_paths: vec![],
            suspicious_ips: vec![],
        }
    }

    fn stub_analysis() -> AnalysisResult {
        AnalysisResult {
            issues: vec![Issue {
                severity: Severity::Info,
                title: "test issue".to_string(),
                explanation: "test explanation".to_string(),
                action: "no action needed".to_string(),
                evidence_indices: vec![],
                evidence: vec![],
            }],
            raw: None,
        }
    }

    async fn mem_store() -> ResultStore {
        ResultStore::new("sqlite::memory:").await.expect("in-memory store must succeed")
    }

    #[tokio::test]
    async fn save_and_get_round_trip() {
        let store = mem_store().await;
        let id = store
            .save("test.log", &stub_summary(), &stub_analysis())
            .await
            .expect("save must succeed");

        let detail = store
            .get(id)
            .await
            .expect("get must succeed")
            .expect("row must exist after save");

        assert_eq!(detail.id, id);
        assert_eq!(detail.filename, "test.log");
        assert_eq!(detail.summary.total, 10);
        assert_eq!(detail.analysis.issues[0].title, "test issue");
    }

    #[tokio::test]
    async fn get_missing_id_returns_none() {
        let store = mem_store().await;
        let result = store.get(9999).await.expect("get must not error");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_returns_newest_first() {
        let store = mem_store().await;
        let id1 = store
            .save("first.log", &stub_summary(), &stub_analysis())
            .await
            .unwrap();
        let id2 = store
            .save("second.log", &stub_summary(), &stub_analysis())
            .await
            .unwrap();

        let rows = store.list().await.expect("list must succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, id2, "newest must be first");
        assert_eq!(rows[1].id, id1);
        assert_eq!(rows[0].filename, "second.log");
    }

    #[test]
    fn parse_timestamp_valid_rfc3339() {
        use chrono::Datelike;
        let dt = parse_timestamp("2026-03-07T09:12:03+00:00".to_string())
            .expect("valid RFC3339 must parse");
        assert_eq!(dt.year(), 2026);
        assert_eq!(dt.month(), 3);
        assert_eq!(dt.day(), 7);
    }

    #[test]
    fn parse_timestamp_rejects_garbage() {
        let dt = parse_timestamp("not a timestamp".to_string());
        assert!(dt.is_err());
    }
}
