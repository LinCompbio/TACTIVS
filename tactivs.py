"""Direct command-line launcher for the published TACTIVS source code."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    source_root = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_root))

    from tactivs.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
