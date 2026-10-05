"""Console entry point: launch the marimo dashboard as an app."""

import sys
from pathlib import Path


def _notebook_path() -> Path:
    return Path(__file__).resolve().parent / "dashboard.py"


def _launch(command: str) -> int:
    """Invoke ``marimo <command> --no-sandbox <notebook>`` forwarding extra args.

    The package's own environment already has the notebook's dependencies, so
    no sandbox is needed (and uv refuses to build one from inside its cache).
    ``--no-sandbox`` also skips marimo's prompt. Any extra command-line
    arguments are forwarded (e.g. ``--port 2718``, ``--headless``).
    """
    from marimo._cli.cli import main as marimo_main

    notebook = _notebook_path()
    sys.argv = ["marimo", command, "--no-sandbox", str(notebook), *sys.argv[1:]]
    return marimo_main()


def main() -> int:
    """Run the dashboard as an app (``marimo run``)."""
    return _launch("run")


def edit() -> int:
    """Open the dashboard in the marimo editor (``marimo edit``)."""
    return _launch("edit")


if __name__ == "__main__":
    raise SystemExit(main())
