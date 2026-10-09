"""The credential routes that bypassed the dynamic URL in the in-tree attempt (PR #108691 review):
``--api-key`` on the command line, and a credential stored with ``hermes auth add``.

Both must yield a NON-EMPTY runtime base URL (core aborts setup with "Provider resolver returned an
empty base URL" otherwise) and still end up at the profile's own account when a client is built.
"""

from __future__ import annotations

import httpx
import pytest

from support import ACCOUNT_A, PROVIDER, TOKEN_A, in_home
from test_routing import OK


def _request_made_with(runtime: dict) -> httpx.Request:
    from providers import get_provider_profile

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    client = get_provider_profile(PROVIDER).create_client(
        api_key=runtime["api_key"], base_url=runtime["base_url"],
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    return seen[0]


def test_explicit_api_key_runtime_has_a_base_url_and_reaches_the_account(make_home):
    from hermes_cli import auth, runtime_provider
    from providers import list_providers

    home = make_home({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})  # no token in .env: it is passed explicitly
    with in_home(home):
        list_providers()  # Hermes scans the bound home's plugins at startup...
        pconfig = auth._registry_lookup(PROVIDER)  # ...and mirrors them into the auth registry
        runtime = runtime_provider._explicit_api_key_provider(
            PROVIDER, pconfig, PROVIDER, {}, TOKEN_A, "", None
        )
        assert runtime["base_url"], "core would abort with 'empty base URL'"
        request = _request_made_with(runtime)
    assert request.url.path == f"/client/v4/accounts/{ACCOUNT_A}/ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TOKEN_A}"


@pytest.mark.parametrize("stored_base_url", [None, ""], ids=["no-base-url", "empty-base-url"])
def test_a_stored_pool_credential_runtime_has_a_base_url_and_reaches_the_account(make_home, stored_base_url):
    from agent.credential_pool import PooledCredential
    from hermes_cli import auth, runtime_provider
    from providers import list_providers

    home = make_home({"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A})
    entry = PooledCredential(
        provider=PROVIDER, id="e1", label="manual", auth_type="api_key", priority=0,
        source="manual", access_token=TOKEN_A, base_url=stored_base_url,
    )
    with in_home(home):
        list_providers()  # startup scan of the bound home's plugins
        assert auth._registry_lookup(PROVIDER) is not None  # the pool path reads the dict directly
        runtime = runtime_provider._resolve_runtime_from_pool_entry(
            provider=PROVIDER, entry=entry, requested_provider=PROVIDER, model_cfg={}
        )
        assert runtime["base_url"], "core would abort with 'empty base URL'"
        request = _request_made_with(runtime)
    assert request.url.path == f"/client/v4/accounts/{ACCOUNT_A}/ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TOKEN_A}"
