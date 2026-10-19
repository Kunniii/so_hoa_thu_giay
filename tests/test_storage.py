"""Tests for storage.py: sandboxing, path safety, and atomic writing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from letter_worker.errors import StorageSecurityError
from letter_worker.storage import SafeStorage


def test_storage_confinement_prevents_escape(tmp_path: Path):
    storage_root = tmp_path / "sandbox"
    storage_root.mkdir()
    storage = SafeStorage(storage_root)

    # Path traversal attempt "../outside.txt"
    with pytest.raises(StorageSecurityError, match="outside storage root"):
        storage.resolve_safe_path("../outside.txt")


def test_storage_resolves_valid_relative_path(tmp_path: Path):
    storage_root = tmp_path / "sandbox"
    storage_root.mkdir()
    storage = SafeStorage(storage_root)

    resolved = storage.resolve_safe_path("sub/file.png")
    assert resolved == (storage_root / "sub/file.png").resolve()


def test_storage_saves_image_atomically(tmp_path: Path):
    storage = SafeStorage(tmp_path)
    img_arr = np.full((120, 120, 3), 200, dtype=np.uint8)
    dest = tmp_path / "out" / "test.png"

    saved = storage.save_image_atomically(img_arr, dest, format_name="PNG")
    assert saved.exists()
    assert saved == dest

    # Verify no dangling temp files left
    temp_files = list(dest.parent.glob(".tmp_*"))
    assert len(temp_files) == 0

    # Verify readable
    with Image.open(saved) as img:
        assert img.size == (120, 120)


def test_storage_saves_json_atomically(tmp_path: Path):
    storage = SafeStorage(tmp_path)
    data = {"status": "ok", "value": 42}
    dest = tmp_path / "meta.json"

    saved = storage.save_json_atomically(data, dest)
    assert saved.exists()
    assert json_reads_correctly(saved, data)


def json_reads_correctly(path: Path, expected: dict) -> bool:
    import json

    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded == expected
