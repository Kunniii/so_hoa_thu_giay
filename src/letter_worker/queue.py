"""SQLite-backed transactional job queue with lease timeouts and progress tracking."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from letter_worker.schemas import (
    JobProgressEvent,
    JobResult,
    LetterJob,
    Point,
    RestoreSettings,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteJobQueue:
    """Manages transactional claim, progress updates, and completion of letter processing jobs."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode by default; manual transactions with BEGIN
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def init_db(self) -> None:
        """Create jobs table and indexes if not already present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT,
                    progress INTEGER DEFAULT 0,
                    message_code TEXT,
                    input_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    output_path TEXT,
                    metadata_path TEXT,
                    corners_json TEXT,
                    settings_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    finished_at TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);"
            )

    def enqueue_job(
        self,
        job: LetterJob,
        max_attempts: int = 3,
    ) -> str:
        """Submit a new job to the queue."""
        corners_data = [p.model_dump() for p in job.corners] if job.corners else None
        settings_data = job.settings.model_dump()
        now = _utc_now_iso()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, stage, progress, message_code,
                    input_path, output_dir, corners_json, settings_json,
                    max_attempts, created_at
                ) VALUES (?, 'queued', 'queued', 0, 'QUEUED', ?, ?, ?, ?, ?, ?);
                """,
                (
                    job.job_id,
                    job.input_path,
                    job.output_dir,
                    json.dumps(corners_data) if corners_data else None,
                    json.dumps(settings_data),
                    max_attempts,
                    now,
                ),
            )
        return job.job_id

    def claim_next_job(self, worker_id: str) -> LetterJob | None:
        """Atomically claim the oldest queued job.

        Returns LetterJob if claimed, or None if queue is empty.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.execute(
                """
                SELECT job_id, input_path, output_dir, corners_json, settings_json, attempts, max_attempts
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1;
                """
            )
            row = cursor.fetchone()
            if not row:
                conn.execute("COMMIT;")
                return None

            job_id = row["job_id"]
            now = _utc_now_iso()

            conn.execute(
                """
                UPDATE jobs
                SET status = 'loading',
                    stage = 'loading',
                    progress = 5,
                    message_code = 'CLAIMED',
                    worker_id = ?,
                    started_at = ?,
                    heartbeat_at = ?,
                    attempts = attempts + 1
                WHERE job_id = ?;
                """,
                (worker_id, now, now, job_id),
            )
            conn.execute("COMMIT;")

            corners = None
            if row["corners_json"]:
                raw_corners = json.loads(row["corners_json"])
                corners = [Point(**pt) for pt in raw_corners]

            settings = RestoreSettings()
            if row["settings_json"]:
                settings = RestoreSettings(**json.loads(row["settings_json"]))

            return LetterJob(
                job_id=job_id,
                input_path=row["input_path"],
                output_dir=row["output_dir"],
                corners=corners,
                settings=settings,
            )
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def update_progress(self, event: JobProgressEvent) -> None:
        """Update job stage, progress, and heartbeat timestamp."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET stage = ?,
                    progress = ?,
                    message_code = ?,
                    heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (event.stage.value, event.progress, event.message_code, now, event.job_id),
            )

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        """Renew lease timestamp for currently active job."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET heartbeat_at = ?
                WHERE job_id = ? AND worker_id = ?;
                """,
                (now, job_id, worker_id),
            )

    def mark_awaiting_corners(
        self,
        job_id: str,
        detected_corners: list[Point] | None,
        confidence: float,
    ) -> None:
        """Flag job as awaiting manual corner verification by user/API."""
        corners_data = [p.model_dump() for p in (detected_corners or [])]
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'awaiting_corners',
                    stage = 'awaiting_corners',
                    progress = 40,
                    message_code = 'AWAITING_CORNER_CONFIRMATION',
                    corners_json = ?,
                    heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (json.dumps(corners_data), now, job_id),
            )

    def confirm_corners(self, job_id: str, corners: list[Point]) -> None:
        """API action: provide confirmed corners and return job to queued state."""
        corners_data = [p.model_dump() for p in corners]
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    stage = 'queued',
                    progress = 0,
                    message_code = 'CORNERS_CONFIRMED',
                    corners_json = ?
                WHERE job_id = ?;
                """,
                (json.dumps(corners_data), job_id),
            )

    def mark_completed(self, job_id: str, result: JobResult) -> None:
        """Mark job successfully completed with persisted file paths."""
        now = _utc_now_iso()
        corners_data = [p.model_dump() for p in (result.detected_corners or [])]
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    stage = 'completed',
                    progress = 100,
                    message_code = 'COMPLETED',
                    output_path = ?,
                    metadata_path = ?,
                    corners_json = ?,
                    finished_at = ?
                WHERE job_id = ?;
                """,
                (result.output_path, result.metadata_path, json.dumps(corners_data), now, job_id),
            )

    def mark_failed(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
        retryable: bool = False,
    ) -> None:
        """Handle job failure, triggering retry or permanent failure."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return

            attempts = row["attempts"]
            max_attempts = row["max_attempts"]

            if retryable and attempts < max_attempts:
                # Requeue for retry
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        stage = 'queued',
                        message_code = 'RETRY_SCHEDULED',
                        error_code = ?,
                        error_message = ?
                    WHERE job_id = ?;
                    """,
                    (error_code, error_message, job_id),
                )
            else:
                # Terminal failure
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        stage = 'failed',
                        progress = 100,
                        message_code = 'FAILED',
                        error_code = ?,
                        error_message = ?,
                        finished_at = ?
                    WHERE job_id = ?;
                    """,
                    (error_code, error_message, now, job_id),
                )

    def reclaim_stale_jobs(self, lease_timeout_seconds: int = 300) -> int:
        """Find in-progress jobs whose worker heartbeat timed out and re-queue or fail them."""
        now = datetime.now(UTC)
        reclaimed_count = 0

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT job_id, status, heartbeat_at, started_at, attempts, max_attempts
                FROM jobs
                WHERE status IN ('loading', 'detecting', 'rectifying', 'restoring', 'quality_check');
                """
            )
            rows = cursor.fetchall()

            for row in rows:
                ts_str = row["heartbeat_at"] or row["started_at"]
                if not ts_str:
                    continue
                try:
                    last_seen = datetime.fromisoformat(ts_str)
                    age_seconds = (now - last_seen).total_seconds()
                except Exception:
                    age_seconds = lease_timeout_seconds + 1

                if age_seconds > lease_timeout_seconds:
                    job_id = row["job_id"]
                    if row["attempts"] < row["max_attempts"]:
                        conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'queued',
                                stage = 'queued',
                                message_code = 'LEASE_EXPIRED_REQUEUED',
                                worker_id = NULL
                            WHERE job_id = ?;
                            """,
                            (job_id,),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'failed',
                                stage = 'failed',
                                message_code = 'LEASE_EXPIRED_FAILED',
                                error_code = 'PROCESSING_TIMEOUT',
                                error_message = 'Worker lease timed out and max attempts reached',
                                finished_at = ?
                            WHERE job_id = ?;
                            """,
                            (_utc_now_iso(), job_id),
                        )
                    reclaimed_count += 1

        return reclaimed_count

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch dictionary representation of job row."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)
