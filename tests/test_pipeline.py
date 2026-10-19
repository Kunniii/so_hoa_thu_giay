"""End-to-end integration tests for LetterProcessor pipeline."""

import json
from pathlib import Path

from letter_worker.config import WorkerConfig
from letter_worker.processor import LetterProcessor
from letter_worker.schemas import (
    JobProgressEvent,
    JobStage,
    LetterJob,
    Point,
    RestoreSettings,
)


def test_pipeline_with_manual_corners(
    temp_storage: Path,
    sample_image_file: Path,
    synthetic_desk_photo,
):
    _, gt_corners = synthetic_desk_photo
    config = WorkerConfig(storage_root=temp_storage)
    processor = LetterProcessor(config)

    progress_history: list[JobProgressEvent] = []

    def on_progress(ev: JobProgressEvent):
        progress_history.append(ev)

    manual_pts = [Point.from_array(pt) for pt in gt_corners]
    job = LetterJob(
        job_id="job_manual_01",
        input_path=sample_image_file.name,
        output_dir="output",
        corners=manual_pts,
        settings=RestoreSettings(preset="pure_bw"),
    )

    result = processor.process(job, on_progress=on_progress)

    assert result.status == "completed"
    assert result.output_path is not None
    assert Path(result.output_path).exists()
    assert result.metadata_path is not None
    assert Path(result.metadata_path).exists()

    # Check metadata content
    with open(result.metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["job_id"] == "job_manual_01"
    assert meta["settings"]["preset"] == "pure_bw"
    assert meta["quality"]["status"] in ("passed", "warning")

    # Verify stage progression
    stages = [ev.stage for ev in progress_history]
    assert JobStage.LOADING in stages
    assert JobStage.DETECTING in stages
    assert JobStage.RECTIFYING in stages
    assert JobStage.RESTORING in stages
    assert JobStage.QUALITY_CHECK in stages
    assert JobStage.COMPLETED in stages


def test_pipeline_with_auto_detection_vintage(
    temp_storage: Path,
    sample_image_file: Path,
):
    config = WorkerConfig(
        storage_root=temp_storage,
        confidence_auto_accept=0.70,  # Synthetic photo detection has ~0.7-0.9 confidence
    )
    processor = LetterProcessor(config)

    job = LetterJob(
        job_id="job_auto_vintage",
        input_path=sample_image_file.name,
        output_dir="output_vintage",
        corners=None,  # Triggers auto-detection
        settings=RestoreSettings(preset="vintage"),
    )

    result = processor.process(job)

    assert result.status == "completed"
    assert result.output_path is not None
    assert Path(result.output_path).exists()
    assert result.confidence is not None
    assert result.confidence >= 0.70


def test_pipeline_triggers_awaiting_corners_when_confidence_insufficient(
    temp_storage: Path,
    sample_image_file: Path,
):
    # Set impossible confidence threshold (0.999)
    config = WorkerConfig(
        storage_root=temp_storage,
        confidence_auto_accept=0.999,
    )
    processor = LetterProcessor(config)

    job = LetterJob(
        job_id="job_needs_corners",
        input_path=sample_image_file.name,
        output_dir="output_pending",
        corners=None,
    )

    result = processor.process(job)

    # Should pause cleanly at awaiting_corners
    assert result.status == "awaiting_corners"
    assert result.output_path is None
    assert result.detected_corners is not None


def test_pipeline_margin_inset(
    temp_storage: Path,
    sample_image_file: Path,
    synthetic_desk_photo,
):
    _, gt_corners = synthetic_desk_photo
    config = WorkerConfig(storage_root=temp_storage)
    processor = LetterProcessor(config)

    manual_pts = [Point.from_array(pt) for pt in gt_corners]
    # Process without margin_inset
    job_full = LetterJob(
        job_id="job_full",
        input_path=sample_image_file.name,
        output_dir="out_full",
        corners=manual_pts,
        settings=RestoreSettings(margin_inset=0.0),
    )
    res_full = processor.process(job_full)

    # Process with 5% margin_inset
    job_inset = LetterJob(
        job_id="job_inset",
        input_path=sample_image_file.name,
        output_dir="out_inset",
        corners=manual_pts,
        settings=RestoreSettings(margin_inset=0.05),
    )
    res_inset = processor.process(job_inset)

    assert res_full.status == "completed"
    assert res_inset.status == "completed"
    assert res_inset.quality.width < res_full.quality.width
    assert res_inset.quality.height < res_full.quality.height
