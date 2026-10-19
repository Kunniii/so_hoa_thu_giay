"""Tests for queue.py: SQLite job lifecycle, concurrency, lease timeouts, and progress."""

from pathlib import Path

import pytest

from letter_worker.queue import SQLiteJobQueue
from letter_worker.schemas import (
    JobProgressEvent,
    JobResult,
    JobStage,
    LetterJob,
    Point,
    RestoreSettings,
)


@pytest.fixture
def queue(tmp_path: Path) -> SQLiteJobQueue:
    db_file = tmp_path / "test_jobs.db"
    return SQLiteJobQueue(db_file)


def test_enqueue_and_claim_job(queue: SQLiteJobQueue):
    job = LetterJob(
        job_id="job_001",
        input_path="input.jpg",
        output_dir="output",
        settings=RestoreSettings(preset="natural"),
    )
    enqueued_id = queue.enqueue_job(job)
    assert enqueued_id == "job_001"

    # Worker 1 claims
    claimed = queue.claim_next_job(worker_id="worker_A")
    assert claimed is not None
    assert claimed.job_id == "job_001"
    assert claimed.settings.preset == "natural"

    # Worker 2 attempts claim on empty queue
    claimed_again = queue.claim_next_job(worker_id="worker_B")
    assert claimed_again is None

    # Check status in DB
    record = queue.get_job("job_001")
    assert record["status"] == "loading"
    assert record["worker_id"] == "worker_A"
    assert record["attempts"] == 1


def test_update_progress_and_heartbeat(queue: SQLiteJobQueue):
    job = LetterJob(job_id="job_progress", input_path="in.jpg", output_dir="out")
    queue.enqueue_job(job)
    queue.claim_next_job(worker_id="w1")

    event = JobProgressEvent(
        job_id="job_progress",
        stage=JobStage.RESTORING,
        progress=70,
        message_code="TEST_PROGRESS",
    )
    queue.update_progress(event)

    rec = queue.get_job("job_progress")
    assert rec["stage"] == "restoring"
    assert rec["progress"] == 70
    assert rec["message_code"] == "TEST_PROGRESS"


def test_awaiting_corners_and_confirmation(queue: SQLiteJobQueue):
    job = LetterJob(job_id="job_corners", input_path="in.jpg", output_dir="out")
    queue.enqueue_job(job)
    queue.claim_next_job(worker_id="w1")

    # Algorithm proposes corners but needs confirmation
    prop_corners = [Point(x=10, y=10), Point(x=100, y=10), Point(x=100, y=200), Point(x=10, y=200)]
    queue.mark_awaiting_corners("job_corners", prop_corners, confidence=0.65)

    rec = queue.get_job("job_corners")
    assert rec["status"] == "awaiting_corners"

    # API receives user confirmation and re-queues
    confirmed = [Point(x=15, y=15), Point(x=95, y=15), Point(x=95, y=195), Point(x=15, y=195)]
    queue.confirm_corners("job_corners", confirmed)

    rec2 = queue.get_job("job_corners")
    assert rec2["status"] == "queued"

    # Worker claims again
    claimed = queue.claim_next_job(worker_id="w2")
    assert claimed is not None
    assert claimed.corners is not None
    assert len(claimed.corners) == 4
    assert claimed.corners[0].x == 15.0


def test_mark_completed(queue: SQLiteJobQueue):
    job = LetterJob(job_id="job_done", input_path="in.jpg", output_dir="out")
    queue.enqueue_job(job)
    queue.claim_next_job("w1")

    res = JobResult(
        job_id="job_done",
        status="completed",
        output_path="/out/res.png",
        metadata_path="/out/meta.json",
        duration_seconds=1.25,
    )
    queue.mark_completed("job_done", res)

    rec = queue.get_job("job_done")
    assert rec["status"] == "completed"
    assert rec["output_path"] == "/out/res.png"
    assert rec["finished_at"] is not None


def test_mark_failed_retry_and_terminal(queue: SQLiteJobQueue):
    job = LetterJob(job_id="job_fail", input_path="in.jpg", output_dir="out")
    queue.enqueue_job(job, max_attempts=2)

    # Attempt 1
    queue.claim_next_job("w1")
    queue.mark_failed("job_fail", "NETWORK_ERR", "Timeout", retryable=True)
    rec1 = queue.get_job("job_fail")
    assert rec1["status"] == "queued"  # Re-queued

    # Attempt 2
    queue.claim_next_job("w2")
    queue.mark_failed("job_fail", "NETWORK_ERR", "Timeout", retryable=True)
    rec2 = queue.get_job("job_fail")
    assert rec2["status"] == "failed"  # Max attempts reached, terminal failure


def test_reclaim_stale_jobs(queue: SQLiteJobQueue):
    job = LetterJob(job_id="stale_job", input_path="in.jpg", output_dir="out")
    queue.enqueue_job(job)
    queue.claim_next_job("worker_dead")

    # Manually backdate heartbeat in DB to simulate dead worker
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET heartbeat_at = '2020-01-01T00:00:00+00:00' WHERE job_id = 'stale_job';"
        )

    reclaimed = queue.reclaim_stale_jobs(lease_timeout_seconds=60)
    assert reclaimed == 1

    rec = queue.get_job("stale_job")
    assert rec["status"] == "queued"
    assert rec["worker_id"] is None
