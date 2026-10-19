"""Worker lifecycle, daemon process, and CLI entry points."""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from letter_worker.config import WorkerConfig, default_config
from letter_worker.processor import LetterProcessor
from letter_worker.queue import SQLiteJobQueue
from letter_worker.schemas import (
    JobProgressEvent,
    LetterJob,
    Point,
    RestoreSettings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("letter_worker")


class WorkerDaemon:
    """Continuous worker daemon that polls for jobs, tracks heartbeats, and invokes processor."""

    def __init__(self, config: WorkerConfig = default_config):
        self.config = config
        self.queue = SQLiteJobQueue(config.database_path)
        self.processor = LetterProcessor(config)
        self._running = True

        # Signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info(f"Received shutdown signal ({signum}), finishing current tasks...")
        self._running = False

    def run(self) -> None:
        logger.info(
            f"Letter Worker started. ID={self.config.worker_id}, DB={self.config.database_path}, Root={self.config.storage_root}"
        )
        last_reclaim_time = 0.0

        while self._running:
            try:
                now = time.time()
                # Periodic stale job reclamation every 60s
                if now - last_reclaim_time > 60.0:
                    reclaimed = self.queue.reclaim_stale_jobs(self.config.job_lease_timeout_seconds)
                    if reclaimed > 0:
                        logger.warning(f"Reclaimed {reclaimed} expired/stale jobs")
                    last_reclaim_time = now

                # Claim next available job
                job = self.queue.claim_next_job(self.config.worker_id)
                if not job:
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                logger.info(f"Claimed job {job.job_id} (input: {job.input_path})")
                self._execute_job(job)

            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(self.config.poll_interval_seconds)

        logger.info("Worker stopped gracefully.")

    def _execute_job(self, job: LetterJob) -> None:
        stop_heartbeat = threading.Event()

        def heartbeat_loop():
            while not stop_heartbeat.wait(timeout=10.0):
                try:
                    self.queue.heartbeat(job.job_id, self.config.worker_id)
                except Exception as ex:
                    logger.debug(f"Heartbeat failed for {job.job_id}: {ex}")

        hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        hb_thread.start()

        def on_progress(event: JobProgressEvent):
            logger.info(
                f"[{job.job_id}] Stage: {event.stage.value.upper()} ({event.progress}%) - {event.message_code}"
            )
            self.queue.update_progress(event)

        try:
            result = self.processor.process(job, on_progress=on_progress)

            if result.status == "completed":
                logger.info(
                    f"Job {job.job_id} COMPLETED in {result.duration_seconds}s -> {result.output_path}"
                )
                self.queue.mark_completed(job.job_id, result)
            elif result.status == "awaiting_corners":
                logger.info(f"Job {job.job_id} AWAITING_CORNERS (confidence: {result.confidence})")
                self.queue.mark_awaiting_corners(
                    job.job_id,
                    result.detected_corners,
                    result.confidence or 0.0,
                )
            else:
                logger.warning(
                    f"Job {job.job_id} FAILED: {result.error_code} - {result.error_message}"
                )
                self.queue.mark_failed(
                    job.job_id,
                    result.error_code or "INTERNAL_ERROR",
                    result.error_message or "Unknown failure",
                    retryable=False,
                )

        finally:
            stop_heartbeat.set()
            hb_thread.join(timeout=1.0)


def cli() -> None:
    """Command-line interface for letter processing."""
    parser = argparse.ArgumentParser(description="Letter Document Processing Worker & CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: worker
    worker_p = subparsers.add_parser("worker", help="Start the worker daemon")
    worker_p.add_argument("--db", default=None, help="Path to SQLite database")
    worker_p.add_argument("--storage", default=None, help="Path to storage root directory")

    # Command: process (one-shot standalone)
    proc_p = subparsers.add_parser("process", help="Process a single image directly without queue")
    proc_p.add_argument("input_path", help="Path to input photo")
    proc_p.add_argument("--output-dir", default="./output", help="Output directory")
    proc_p.add_argument("--preset", choices=["pure_bw", "natural", "vintage"], default="natural")
    proc_p.add_argument("--denoise", choices=["off", "light", "medium"], default="light")
    proc_p.add_argument("--threshold-method", choices=["adaptive", "otsu"], default="adaptive")
    proc_p.add_argument("--threshold-strength", type=float, default=0.5)
    proc_p.add_argument("--corners", help="JSON string or file of 4 corners [[x,y],...]")
    proc_p.add_argument("--bleed-through", type=float, default=0.82, help="Bleed-through reduction threshold (0.50-0.98)")
    proc_p.add_argument("--margin-inset", type=float, default=0.0, help="Fraction to crop inward from edges (0.0-0.10)")

    # Command: enqueue
    enq_p = subparsers.add_parser("enqueue", help="Add a job to SQLite queue")
    enq_p.add_argument("job_id", help="Unique job ID")
    enq_p.add_argument("input_path", help="Input image path")
    enq_p.add_argument("--output-dir", default="./output", help="Output directory")
    enq_p.add_argument("--preset", choices=["pure_bw", "natural", "vintage"], default="natural")
    enq_p.add_argument("--db", default=None, help="Database path")

    # Command: status
    stat_p = subparsers.add_parser("status", help="Check status of a job in DB")
    stat_p.add_argument("job_id", help="Job ID")
    stat_p.add_argument("--db", default=None, help="Database path")

    # Command: init-db
    init_p = subparsers.add_parser("init-db", help="Initialize SQLite database")
    init_p.add_argument("--db", default=None, help="Database path")

    args = parser.parse_args()

    config = WorkerConfig()
    if hasattr(args, "storage") and args.storage:
        config.storage_root = Path(args.storage).resolve()
    if hasattr(args, "db") and args.db:
        config.database_path = Path(args.db).resolve()

    if args.command == "worker":
        daemon = WorkerDaemon(config)
        daemon.run()

    elif args.command == "init-db":
        q = SQLiteJobQueue(config.database_path)
        print(f"Database initialized at: {q.db_path}")

    elif args.command == "process":
        processor = LetterProcessor(config)
        corners_list = None
        if args.corners:
            raw = json.loads(args.corners)
            corners_list = [Point(x=pt[0], y=pt[1]) for pt in raw]

        job = LetterJob(
            job_id=f"cli_{int(time.time())}",
            input_path=args.input_path,
            output_dir=args.output_dir,
            corners=corners_list,
            settings=RestoreSettings(
                preset=args.preset,
                denoise=args.denoise,
                threshold_method=args.threshold_method,
                threshold_strength=args.threshold_strength,
                bleed_through_reduction=args.bleed_through,
                margin_inset=args.margin_inset,
            ),
        )

        def log_progress(event: JobProgressEvent):
            print(f"[{event.progress}%] {event.stage.value.upper()}: {event.message_code}")

        res = processor.process(job, on_progress=log_progress)
        print("\nProcessing Result:")
        print(json.dumps(res.model_dump(), indent=2))
        if res.status != "completed":
            sys.exit(1)

    elif args.command == "enqueue":
        q = SQLiteJobQueue(config.database_path)
        job = LetterJob(
            job_id=args.job_id,
            input_path=args.input_path,
            output_dir=args.output_dir,
            settings=RestoreSettings(preset=args.preset),
        )
        q.enqueue_job(job)
        print(f"Job '{args.job_id}' enqueued successfully.")

    elif args.command == "status":
        q = SQLiteJobQueue(config.database_path)
        j = q.get_job(args.job_id)
        if not j:
            print(f"Job '{args.job_id}' not found.")
            sys.exit(1)
        print(json.dumps(j, indent=2))


if __name__ == "__main__":
    cli()
