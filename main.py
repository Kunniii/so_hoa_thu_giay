import sys
from pathlib import Path

# Ensure src/ is on python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from letter_worker.main import cli  # noqa: E402

if __name__ == "__main__":
    cli()
