"""Where a request and its token actually go, with several Hermes profiles in ONE process.

Hermes keeps a single process-wide registry row per provider name and overwrites its base URL from
the last home to load. These tests exercise the real credential path and make real client calls
against an in-memory transport, so they assert the HTTP boundary, not an intermediate value.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from support import ACCOUNT_A, ACCOUNT_B, PROVIDER, TOKEN_A, TOKEN_B, in_home

OK = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}],
}


def call(home, extra_kwargs=None):
    """One chat call the way Hermes makes it: resolve credentials, then ask the profile for a client.

    Returns the list of requests that reached the (fake) network, and the client.
    """
    from hermes_cli import auth
    from providers import get_provider_profile, list_providers

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    with in_home(home):
        list_providers()  # Hermes scans a home's plugins when it first uses that home
        creds = auth.resolve_api_key_provider_credentials(PROVIDER)
        profile = get_provider_profile(PROVIDER)
        kwargs = {
            "api_key": creds["api_key"],
            "base_url": creds["base_url"],
            "http_client": httpx.Client(transport=httpx.MockTransport(handler)),
            **(extra_kwargs or {}),
        }
        client = profile.create_client(**kwargs)
        client.chat.completions.create(model="@cf/test/model", messages=[{"role": "user", "content": "hi"}])
    return seen, client


def test_registered_with_hermes_and_url_carries_no_account(make_home):
    from hermes_cli import auth
    from providers import get_provider_profile, provider_source

    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    with in_home(home):
        profile = get_provider_profile(PROVIDER)
        assert profile is not None and provider_source(PROVIDER) == "user"
        assert get_provider_profile("cf-workers-ai") is profile
        # Hermes mirrors profiles into a SEPARATE auth registry; a profile can resolve while the
        # registry silently skips it (e.g. empty env_vars), so assert the registry too.
        assert auth._registry_lookup(PROVIDER) is not None
        row = auth.PROVIDER_REGISTRY[PROVIDER]
        assert ACCOUNT_A not in row.inference_base_url
        assert row.inference_base_url.startswith("https://api.cloudflare.com/")


def test_each_profile_reaches_its_own_account_in_one_process(make_home):
    a = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    b = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_B, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_B})

    for home, account, token in [(a, ACCOUNT_A, TOKEN_A), (b, ACCOUNT_B, TOKEN_B), (a, ACCOUNT_A, TOKEN_A)]:
        seen, _ = call(home)
        assert len(seen) == 1
        assert seen[0].url.host == "api.cloudflare.com"
        assert seen[0].url.path == f"/client/v4/accounts/{account}/ai/v1/chat/completions"
        assert seen[0].headers["authorization"] == f"Bearer {token}"


def test_a_proxy_override_belongs_to_its_profile_only(make_home):
    plain = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    proxied = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_B, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_B,
        "CLOUDFLARE_BASE_URL": "https://proxy.example.test/ai/v1",
    })

    for _ in range(2):  # alternate: the shared registry row must not remember the other profile
        seen_proxied, _c = call(proxied)
        assert seen_proxied[0].url.host == "proxy.example.test"
        assert seen_proxied[0].headers["authorization"] == f"Bearer {TOKEN_B}"
        seen_plain, _c = call(plain)
        assert seen_plain[0].url.host == "api.cloudflare.com"
        assert seen_plain[0].headers["authorization"] == f"Bearer {TOKEN_A}"


def test_an_account_change_applies_without_a_restart(make_home):
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    assert f"/accounts/{ACCOUNT_A}/" in call(home)[0][0].url.path

    (home / ".env").write_text(
        f"CLOUDFLARE_API_TOKEN={TOKEN_A}\nCLOUDFLARE_ACCOUNT_ID={ACCOUNT_B}\n# changed\n"
    )
    assert f"/accounts/{ACCOUNT_B}/" in call(home)[0][0].url.path


def test_an_explicit_base_url_beats_the_account(make_home):
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    seen, _ = call(home, {"base_url": "https://gateway.example.test/v1"})
    assert seen[0].url.host == "gateway.example.test"


def test_token_env_order_prefers_api_token_over_api_key(make_home):
    both = make_home({
        "CLOUDFLARE_API_KEY": TOKEN_B, "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A,
    })
    legacy = make_home({"CLOUDFLARE_API_KEY": TOKEN_B, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    assert call(both)[0][0].headers["authorization"] == f"Bearer {TOKEN_A}"
    assert call(legacy)[0][0].headers["authorization"] == f"Bearer {TOKEN_B}"


@pytest.mark.parametrize("bad", ["", "../../evil", "a b", "x" * 80])
def test_a_missing_or_malformed_account_fails_clearly_and_sends_nothing(make_home, bad):
    env = {"CLOUDFLARE_API_TOKEN": TOKEN_A}
    if bad:
        env["CLOUDFLARE_ACCOUNT_ID"] = f'"{bad}"'
    home = make_home(env)

    from hermes_cli import auth
    from providers import get_provider_profile, list_providers

    sent: list[httpx.Request] = []
    with in_home(home):
        list_providers()
        creds = auth.resolve_api_key_provider_credentials(PROVIDER)
        client = get_provider_profile(PROVIDER).create_client(
            api_key=creds["api_key"], base_url=creds["base_url"],
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json=OK))),
        )
        with pytest.raises(openai.BadRequestError) as caught:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert "CLOUDFLARE_ACCOUNT_ID" in str(caught.value)
    assert sent == []  # the caller's transport was never used: nothing left the machine


def test_sdk_retries_are_off_so_the_conversation_loop_owns_backoff(make_home):
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    _, client = call(home)
    assert client.max_retries == 0
