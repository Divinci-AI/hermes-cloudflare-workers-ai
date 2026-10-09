"""Fixtures. Environment isolation lives in ``support`` and must be imported first."""

from __future__ import annotations

import shutil

import pytest

from support import PLUGIN_FILES, PROVIDER, REPO  # noqa: F401  (import order: isolates HERMES_HOME)


@pytest.fixture
def make_home(tmp_path):
    """Build a Hermes home with the plugin installed and the given ``.env`` entries."""
    count = 0

    def build(env: dict[str, str]):
        nonlocal count
        count += 1
        home = tmp_path / f"home{count}"
        plugin_dir = home / "plugins" / "model-providers" / PROVIDER
        plugin_dir.mkdir(parents=True)
        for name in PLUGIN_FILES:
            shutil.copy2(REPO / name, plugin_dir / name)
        (home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
        return home

    return build
