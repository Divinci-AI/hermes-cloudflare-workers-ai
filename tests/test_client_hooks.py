"""``create_client`` replaces core's own client setup, so it must repeat the parts that matter."""

from __future__ import annotations

import sys
import types

from support import ACCOUNT_A, PROVIDER, TOKEN_A, in_home


def _build(make_home, monkeypatch, *, launched_by_hermes: bool):
    from providers import get_provider_profile, list_providers

    recorded = {}
    sentinel = object()
    fake = types.ModuleType("agent.process_bootstrap")
    fake.OpenAI = lambda **kwargs: recorded.setdefault("kwargs", kwargs) and types.SimpleNamespace(**kwargs)
    fake.build_keepalive_http_client = lambda base_url, **kw: recorded.setdefault("keepalive_for", base_url) and sentinel
    monkeypatch.setitem(sys.modules, "agent.process_bootstrap", fake)
    if launched_by_hermes:  # the launcher has already imported hermes_bootstrap; stand-in for that
        monkeypatch.setitem(sys.modules, "hermes_bootstrap", types.ModuleType("hermes_bootstrap"))
    home = make_home({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    with in_home(home):
        list_providers()
        client = get_provider_profile(PROVIDER).create_client(api_key=TOKEN_A, base_url="https://api.cloudflare.com/client/v4/ai/v1")
    return client, recorded, sentinel


def test_under_the_hermes_launcher_it_uses_core_helpers(make_home, monkeypatch):
    client, recorded, sentinel = _build(make_home, monkeypatch, launched_by_hermes=True)
    assert recorded["kwargs"]["http_client"] is sentinel  # keepalive client, like core's own path
    assert recorded["kwargs"]["max_retries"] == 0
    assert recorded["keepalive_for"].startswith(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_A}/")


def test_outside_the_launcher_it_never_imports_hermes_bootstrap_machinery(make_home, monkeypatch):
    client, recorded, _ = _build(make_home, monkeypatch, launched_by_hermes=False)
    assert "kwargs" not in recorded, "core helpers were used although Hermes did not launch this process"
    assert str(client.base_url).startswith(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_A}/")
