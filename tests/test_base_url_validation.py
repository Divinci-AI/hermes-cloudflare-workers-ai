"""Where the bearer token may be sent when a base URL is overridden.

The token goes wherever the resolved URL points. These tests pin the three ways it could be sent
somewhere it should not: a URL taken from the PROCESS environment, a URL that is not https, and a URL
with credentials embedded in it. Each is asserted at the HTTP boundary (what request, to which host),
not on an intermediate value.
"""

from __future__ import annotations

import importlib.util

import httpx
import openai
import pytest

from support import ACCOUNT_A, PROVIDER, REPO, TOKEN_A, in_home
from test_routing import OK, call

_spec = importlib.util.spec_from_file_location("cfwai_endpoint", REPO / "endpoint.py")
endpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(endpoint)

PASSWORD = "hunter2-do-not-leak"

ACCEPTED = [
    ("https://proxy.example.test/ai/v1", "https://proxy.example.test/ai/v1"),
    ("https://proxy.example.test/ai/v1/", "https://proxy.example.test/ai/v1"),  # trailing slash stripped
    ("https://gateway.ai.cloudflare.com/v1/acct/gw/workers-ai/v1", "https://gateway.ai.cloudflare.com/v1/acct/gw/workers-ai/v1"),
    ("https://proxy.example.test:8443/v1", "https://proxy.example.test:8443/v1"),
    ("http://localhost:8787/v1", "http://localhost:8787/v1"),
    ("http://127.0.0.1:8787/v1", "http://127.0.0.1:8787/v1"),
    ("http://[::1]:8787/v1", "http://[::1]:8787/v1"),
    ("HTTP://LOCALHOST:8787/v1", "HTTP://LOCALHOST:8787/v1"),
]

REJECTED = [
    ("http://evil.example.test/v1", "plain http"),
    ("http://localhost.evil.example.test/v1", "plain http"),  # a look-alike of a loopback name
    ("http://127.0.0.1.evil.example.test/v1", "plain http"),
    ("http://10.0.0.1:8080/v1", "plain http"),  # private is not loopback
    (f"https://user:{PASSWORD}@proxy.example.test/v1", "embedded credentials"),
    ("https://user@proxy.example.test/v1", "embedded credentials"),
    (f"http://user:{PASSWORD}@127.0.0.1:8787/v1", "embedded credentials"),  # loopback http still no userinfo
    ("https://proxy.example.test/v1?key=abc", "query"),
    ("https://proxy.example.test/v1#frag", "query"),
    ("ftp://proxy.example.test/v1", "https://"),
    ("javascript:alert(1)", "no host"),
    ("file:///etc/passwd", "no host"),
    ("https://", "no host"),
    ("//proxy.example.test/v1", "https://"),  # scheme-relative: has a host, no scheme
    ("https://proxy.example.test:notaport/v1", "valid URL"),
    ("https://proxy .example.test/v1", "whitespace"),
    ("https://proxy.example.test/v1\r\nX-Evil: 1", "whitespace"),
]


@pytest.mark.parametrize("url,expected", ACCEPTED)
def test_validate_accepts_https_and_loopback_http(url, expected):
    assert endpoint.validate_base_url(url, source="X") == expected


@pytest.mark.parametrize("url,reason", REJECTED)
def test_validate_rejects_unsafe_urls_without_echoing_them(url, reason):
    with pytest.raises(endpoint.WorkersAIConfigError) as caught:
        endpoint.validate_base_url(url, source="CLOUDFLARE_BASE_URL")
    message = str(caught.value)
    assert reason in message
    assert PASSWORD not in message and "evil.example" not in message  # the value is never repeated
    assert "CLOUDFLARE_BASE_URL" in message


# ---- the process environment must never choose where the token goes ------------------------------

def test_a_base_url_exported_in_the_shell_does_not_redirect_the_token(make_home, monkeypatch):
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "http://evil.example.test/v1")
    seen, _ = call(home)
    assert len(seen) == 1
    assert seen[0].url.host == "api.cloudflare.com" and seen[0].url.scheme == "https"
    assert seen[0].url.path == f"/client/v4/accounts/{ACCOUNT_A}/ai/v1/chat/completions"


def test_even_a_valid_https_shell_override_is_ignored_the_profile_env_decides(make_home, monkeypatch):
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://shell-proxy.example.test/v1")
    assert call(home)[0][0].url.host == "api.cloudflare.com"


def test_the_profile_env_override_still_works(make_home, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://shell-proxy.example.test/v1")  # must lose to .env
    home = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A,
        "CLOUDFLARE_BASE_URL": "https://proxy.example.test/ai/v1",
    })
    seen, _ = call(home)
    assert seen[0].url.host == "proxy.example.test"
    assert seen[0].headers["authorization"] == f"Bearer {TOKEN_A}"


def test_a_loopback_http_override_works_end_to_end(make_home):
    home = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A,
        "CLOUDFLARE_BASE_URL": "http://127.0.0.1:8787/v1",
    })
    seen, _ = call(home)
    assert (seen[0].url.scheme, seen[0].url.host, seen[0].url.port) == ("http", "127.0.0.1", 8787)


# ---- an unsafe URL fails clearly and sends nothing -------------------------------------------------

def _refuses(home, **create_kwargs):
    """Make a call; return (requests that left, the BadRequestError message)."""
    from hermes_cli import auth
    from providers import get_provider_profile, list_providers

    sent: list[httpx.Request] = []
    with in_home(home):
        list_providers()
        creds = auth.resolve_api_key_provider_credentials(PROVIDER)
        client = get_provider_profile(PROVIDER).create_client(**{
            "api_key": creds["api_key"], "base_url": creds["base_url"],
            "http_client": httpx.Client(transport=httpx.MockTransport(
                lambda r: sent.append(r) or httpx.Response(200, json=OK))),
            **create_kwargs,
        })
        with pytest.raises(openai.BadRequestError) as caught:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    return sent, str(caught.value)


@pytest.mark.parametrize("bad", [
    "http://evil.example.test/v1",
    f"https://user:{PASSWORD}@proxy.example.test/v1",
])
def test_an_unsafe_profile_override_is_refused_and_the_token_is_never_sent(make_home, bad):
    home = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A, "CLOUDFLARE_BASE_URL": bad,
    })
    sent, message = _refuses(home)
    assert sent == []
    assert "CLOUDFLARE_BASE_URL" in message
    assert PASSWORD not in message and "evil.example" not in message


@pytest.mark.parametrize("bad", ["http://evil.example.test/v1", f"https://u:{PASSWORD}@x.example.test/v1"])
def test_an_unsafe_caller_supplied_base_url_is_refused_too(make_home, bad):
    """core hands in model.base_url / a pooled credential's stored URL; they get the same checks."""
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    sent, message = _refuses(home, base_url=bad)
    assert sent == []
    assert PASSWORD not in message and "evil.example" not in message


def test_the_catalog_never_sends_the_key_to_an_unsafe_override(make_home):
    from providers import get_provider_profile, list_providers

    home = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A,
        "CLOUDFLARE_BASE_URL": "http://evil.example.test/v1",
    })
    with in_home(home):
        list_providers()
        profile = get_provider_profile(PROVIDER)
        assert profile.fetch_models(api_key=TOKEN_A) is None
        assert profile.fetch_models(api_key=TOKEN_A, base_url="http://evil.example.test/v1") is None


# ---- account id: fullmatch, no trailing-newline slip ----------------------------------------------

def test_account_id_pattern_is_anchored_at_both_ends():
    assert endpoint._SAFE_ACCOUNT_ID.fullmatch("a" * 32)
    assert not endpoint._SAFE_ACCOUNT_ID.fullmatch("a" * 32 + "\n")
    assert not endpoint._SAFE_ACCOUNT_ID.fullmatch("abc.def")
    assert not endpoint._SAFE_ACCOUNT_ID.fullmatch("a/b")
