"""Catalog parsing and paging (pure), then the profile's ``fetch_models`` over a fake network."""

from __future__ import annotations

import json
import sys

import pytest

from support import ACCOUNT_A, PROVIDER, REPO, TOKEN_A, in_home

sys.path.insert(0, str(REPO))  # the plugin root is a package; load its pure modules by file path
import importlib.util  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"cfwai_{name}", REPO / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load("catalog")


def row(name, task="Text Generation", **extra):
    return {"id": "0f9d3c1e-1a2b-4c3d-8e4f-123456789abc", "name": name, "task": {"name": task}, **extra}


def test_keeps_chat_models_and_drops_the_rest():
    page = {"result": [
        row("@cf/meta/llama-3.1-8b-instruct"),
        row("@cf/qwen/qwen3.8-27b", task="Image-Text-to-Text"),  # multimodal chat: must survive
        row("@cf/black-forest-labs/flux-1-schnell", task="Text-to-Image"),
        row("@cf/baai/bge-large-en-v1.5", task="Text Embeddings"),
        row("@cf/openai/whisper", task="Automatic Speech Recognition"),
    ]}
    slugs, _ = catalog.parse_page(page)
    assert slugs == ["@cf/meta/llama-3.1-8b-instruct", "@cf/qwen/qwen3.8-27b"]


def test_a_slug_is_recognised_by_prefix_not_by_its_length():
    # 36 characters with four hyphens - the exact shape of a UUID, and a real model slug.
    lookalike = "@cf/deepseek-ai/deepseek-v4-pro-0813"
    assert len(lookalike) == 36 and lookalike.count("-") == 4
    slugs, _ = catalog.parse_page({"result": [row(lookalike)]})
    assert slugs == [lookalike]


def test_a_row_with_only_an_internal_uuid_is_dropped():
    uuid_only = {"id": "0f9d3c1e-1a2b-4c3d-8e4f-123456789abc", "name": "friendly name", "task": {"name": "Text Generation"}}
    assert catalog.parse_page({"result": [uuid_only]})[0] == []


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"result": None}, {"result": "no"}, {"result": {"x": 1}}])
def test_malformed_payloads_yield_nothing_instead_of_raising(payload):
    assert catalog.parse_page(payload)[0] == []


def test_task_may_be_a_plain_string():
    assert catalog.parse_page({"result": [{"name": "@cf/a/b", "task": "Text Generation"}]})[0] == ["@cf/a/b"]


def _pager(pages, total_count=None, per_page=50):
    calls = []

    def fetch(url):
        calls.append(url)
        n = len(calls)
        if n > len(pages):
            return {"result": []}
        info = {"page": n, "per_page": per_page}
        if total_count is not None:
            info["total_count"] = total_count
        return {"result": pages[n - 1], "result_info": info}

    return fetch, calls


def test_pages_until_the_reported_total_and_dedupes():
    fetch, calls = _pager([[row("@cf/a/one"), row("@cf/a/two")], [row("@cf/a/two"), row("@cf/a/three")]], total_count=100)
    assert catalog.fetch_chat_models(fetch, ACCOUNT_A) == ["@cf/a/one", "@cf/a/two", "@cf/a/three"]
    assert len(calls) == 2  # 100 rows at 50 per page -> two pages, no third request
    assert f"/accounts/{ACCOUNT_A}/ai/models/search?per_page=50&page=1" in calls[0]


def test_stops_on_an_empty_page_when_no_total_is_reported():
    fetch, calls = _pager([[row("@cf/a/one")]])
    assert catalog.fetch_chat_models(fetch, ACCOUNT_A) == ["@cf/a/one"]
    assert len(calls) == 2


def test_a_runaway_catalog_is_bounded():
    calls = []

    def endless(url):
        calls.append(url)
        return {"result": [row(f"@cf/a/m{len(calls)}")]}

    models = catalog.fetch_chat_models(endless, ACCOUNT_A)
    assert len(calls) == catalog.MAX_PAGES and len(models) == catalog.MAX_PAGES


def test_a_failed_first_page_is_none_but_a_later_failure_keeps_what_we_have():
    def broken(url):
        raise OSError("down")

    assert catalog.fetch_chat_models(broken, ACCOUNT_A) is None

    state = {"n": 0}

    def flaky(url):
        state["n"] += 1
        if state["n"] > 1:
            raise OSError("down")
        return {"result": [row("@cf/a/one")], "result_info": {"per_page": 50, "total_count": 500}}

    assert catalog.fetch_chat_models(flaky, ACCOUNT_A) == ["@cf/a/one"]


# ---- the profile's fetch_models over a faked network ----------------------------------------


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def network(monkeypatch):
    """Replace the credentialed opener Hermes's profiles use; record each request."""
    seen = []
    routes = {}

    def opener(request, timeout=None):
        seen.append(request)
        for fragment, body in routes.items():
            if fragment in request.full_url:
                return FakeResponse(body)
        raise OSError(f"no route for {request.full_url}")

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)
    return seen, routes


def _profile_for(home):
    from providers import get_provider_profile, list_providers

    list_providers()
    return get_provider_profile(PROVIDER)


def test_fetch_models_queries_the_bound_accounts_search_endpoint(make_home, network):
    seen, routes = network
    routes["/ai/models/search"] = {"result": [row("@cf/a/one")], "result_info": {"per_page": 50, "total_count": 1}}
    home = make_home({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    with in_home(home):
        models = _profile_for(home).fetch_models(api_key=TOKEN_A, base_url="https://api.cloudflare.com/client/v4/ai/v1")
    assert models == ["@cf/a/one"]
    assert seen[0].full_url.startswith(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_A}/ai/models/search")
    assert seen[0].get_header("Authorization") == f"Bearer {TOKEN_A}"


def test_fetch_models_prefers_a_proxy_catalog_when_one_is_configured(make_home, network):
    seen, routes = network
    routes["proxy.example.test/ai/v1/models"] = {"data": [{"id": "@cf/proxy/only"}]}
    home = make_home({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A, "CLOUDFLARE_BASE_URL": "https://proxy.example.test/ai/v1"})
    with in_home(home):
        models = _profile_for(home).fetch_models(api_key=TOKEN_A, base_url="https://api.cloudflare.com/client/v4/ai/v1")
    assert models == ["@cf/proxy/only"]
    assert all("api.cloudflare.com" not in r.full_url for r in seen)  # the public endpoint was never asked


@pytest.mark.parametrize("env,key", [({}, TOKEN_A), ({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A}, None)],
                         ids=["no-account", "no-key"])
def test_fetch_models_makes_no_request_without_an_account_or_a_key(make_home, network, env, key):
    seen, _ = network
    home = make_home(env)
    with in_home(home):
        assert _profile_for(home).fetch_models(api_key=key) is None
    assert seen == []


def test_a_cached_catalog_is_reused_not_refetched(make_home, network):
    seen, routes = network
    routes["/ai/models/search"] = {"result": [row("@cf/a/one"), row("@cf/deepseek-ai/deepseek-v4-pro-0813")],
                                   "result_info": {"per_page": 50, "total_count": 2}}
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    from hermes_cli.models import cached_provider_model_ids

    with in_home(home):
        _profile_for(home)
        first = cached_provider_model_ids(PROVIDER)
        fetched_after_first = len(seen)
        second = cached_provider_model_ids(PROVIDER)
    assert "@cf/a/one" in first and first == second
    assert fetched_after_first >= 1
    assert len(seen) == fetched_after_first, "second call refetched: the cache was bypassed"
