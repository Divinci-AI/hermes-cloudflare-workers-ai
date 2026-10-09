"""Shared test support: load the plugin through Hermes's REAL discovery, inside throwaway Hermes homes.

Nothing here may read the real ``~/.hermes``: the process home is pointed at a scratch dir BEFORE any
hermes module is imported, and every CLOUDFLARE_* variable is removed from the process env so a
value from the developer's shell cannot leak into an assertion.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HERMES_SRC = Path(os.environ.get("HERMES_AGENT_DIR", Path.home() / ".hermes" / "hermes-agent"))

_PROCESS_HOME = Path(tempfile.mkdtemp(prefix="hermes-cfwai-process-home-"))
os.environ["HERMES_HOME"] = str(_PROCESS_HOME)
for _name in [n for n in os.environ if n.startswith("CLOUDFLARE_")]:
    del os.environ[_name]

# Importing hermes_bootstrap (directly, or via agent.process_bootstrap) runs Hermes's launch
# preparation, which on 2026-10-08 rebuilt the LIVE checkout's UI/TUI/desktop assets and then hung on a
# network dependency sync. A ``None`` entry makes any such import raise instead of running it.
sys.modules["hermes_bootstrap"] = None

if not (HERMES_SRC / "providers" / "__init__.py").exists():
    pytest.exit(f"hermes-agent source not found at {HERMES_SRC}; set HERMES_AGENT_DIR", returncode=2)
sys.path.insert(0, str(HERMES_SRC))

PROVIDER = "cloudflare-workers-ai"
PLUGIN_FILES = ("__init__.py", "plugin.yaml", "endpoint.py", "catalog.py", "client.py")

# Fake, obviously non-secret values, assembled at runtime.
ACCOUNT_A = "a" * 32
ACCOUNT_B = "b" * 32
TOKEN_A = "cfut_" + "A" * 40
TOKEN_B = "cfut_" + "B" * 40



@contextmanager
def in_home(home: Path):
    """Bind *home* for the current context, exactly as a routed profile would be."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)
