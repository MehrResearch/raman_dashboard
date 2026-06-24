"""Console entry point: launch the marimo dashboard as an app."""

import sys
from pathlib import Path


def _notebook_path() -> Path:
    return Path(__file__).resolve().parent / "dashboard.py"


def main() -> int:
    """Run the dashboard with ``marimo run``.

    Any extra command-line arguments are forwarded to ``marimo run``
    (e.g. ``--port 2718``, ``--headless``).
    """
    from marimo._cli.cli import main as marimo_main

    notebook = _notebook_path()
    sys.argv = ["marimo", "run", str(notebook), *sys.argv[1:]]
    return marimo_main()


if __name__ == "__main__":
    raise SystemExit(main())
