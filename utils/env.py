"""Load this checkout's .env — and only this checkout's.

`load_dotenv()` with no path resolves the file by walking UP from the calling
module. Run from a git worktree under .claude/worktrees/<name>/, that walk
leaves the worktree and finds the parent checkout's .env, so a test process
came up holding production Alpaca keys and the live Telegram token. On
2026-08-03 15:24 that is how `pytest tests/` messaged the operator's phone.

Anchoring on the repo root makes a worktree without its own .env come up with
no credentials, which is the correct and safe result: README provisions .env at
the repo root, so real runs are unaffected.
"""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"


def load_project_env(override: bool = True) -> bool:
    """Load PROJECT_ENV. Returns False when this checkout has no .env."""
    return load_dotenv(dotenv_path=PROJECT_ENV, override=override)
