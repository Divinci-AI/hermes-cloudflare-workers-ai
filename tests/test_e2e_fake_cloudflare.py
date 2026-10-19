"""End to end, over real sockets: the plugin loaded through Hermes's real discovery from a throwaway
HERMES_HOME, the real OpenAI SDK, and a fake Cloudflare (``fakecf``) on 127.0.0.1.

Needs no Cloudflare credentials and no network. What it adds over the unit tests: real streaming
framing, real response headers (``Retry-After``), real HTTP status handling in the SDK, and the verdict
Hermes's own classifier reaches for each failure - the thing that decides whether a turn retries,
rotates the credential, or gives up.
"""

from __future__ import annotations

import httpx  # noqa: F401  (imported so a missing dependency fails at collection, not mid-test)
import openai
import pytest

from fakecf import MODELS, FakeCloudflare
from support import ACCOUNT_A, PROVIDER, TOKEN_A, in_home

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def cloudflare():
    server = FakeCloudflare()
    yield server
    server.close()


@pytest.fixture
def stack(make_home, cloudflare):
    """(profile, client, cloudflare) with the plugin bound to a home whose override is the fake."""
    from hermes_cli import auth
    from providers import get_provider_profile, list_providers

    cloudflare.seen.clear()
    home = make_home({
        "CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_A,
        "CLOUDFLARE_BASE_URL": cloudflare.base_url,
    })
    with in_home(home):
        list_providers()
        creds = auth.resolve_api_key_provider_credentials(PROVIDER)
        profile = get_provider_profile(PROVIDER)
        client = profile.create_client(api_key=creds["api_key"], base_url=creds["base_url"])
        yield profile, client, cloudflare


def _classify(error, cloudflare):
    from agent.error_classifier import classify_api_error

    return classify_api_error(error, provider=PROVIDER, model="@cf/test/x", base_url=cloudflare.base_url, api_key=TOKEN_A)


def test_a_completion_reaches_the_server_with_the_bearer_token(stack):
    _, client, cloudflare = stack
    reply = client.chat.completions.create(model="@cf/test/ok", messages=MESSAGES)
    assert reply.choices[0].message.content == "ok"
    (seen,) = cloudflare.posts()
    assert seen.path == "/client/v4/accounts/e2e/ai/v1/chat/completions"
    assert seen.authorization == f"Bearer {TOKEN_A}"
    assert seen.body["model"] == "@cf/test/ok"


def test_streaming_is_reassembled(stack):
    _, client, cloudflare = stack
    stream = client.chat.completions.create(model="@cf/test/ok", messages=MESSAGES, stream=True)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "hello"
    assert len(cloudflare.posts()) == 1


@pytest.mark.parametrize("scenario,error_type,status", [
    ("e401", openai.AuthenticationError, 401),
    ("e403", openai.PermissionDeniedError, 403),
    ("e429-capacity", openai.RateLimitError, 429),
    ("e429-quota", openai.RateLimitError, 429),
    ("e500", openai.InternalServerError, 500),
])
def test_failures_surface_once_and_never_contain_the_token(stack, scenario, error_type, status):
    _, client, cloudflare = stack
    with pytest.raises(error_type) as caught:
        client.chat.completions.create(model=f"@cf/test/{scenario}", messages=MESSAGES)
    assert caught.value.status_code == status
    assert len(cloudflare.posts()) == 1, "the SDK must not retry: the conversation loop owns backoff"
    assert TOKEN_A not in str(caught.value) and TOKEN_A not in repr(caught.value)


def test_the_capacity_429_carries_retry_after_and_is_classified_overloaded(stack):
    _, client, cloudflare = stack
    with pytest.raises(openai.RateLimitError) as caught:
        client.chat.completions.create(model="@cf/test/e429-capacity", messages=MESSAGES)
    assert caught.value.response.headers["retry-after"] == "7"  # what Hermes's backoff reads
    verdict = _classify(caught.value, cloudflare)
    assert verdict.reason.name == "overloaded"
    assert verdict.retryable is True and verdict.should_rotate_credential is False


def test_an_ordinary_429_is_still_a_rate_limit_that_rotates(stack):
    _, client, cloudflare = stack
    with pytest.raises(openai.RateLimitError) as caught:
        client.chat.completions.create(model="@cf/test/e429-quota", messages=MESSAGES)
    verdict = _classify(caught.value, cloudflare)
    assert verdict.reason.name == "rate_limit" and verdict.should_rotate_credential is True


def test_an_auth_failure_is_classified_as_auth(stack):
    _, client, cloudflare = stack
    with pytest.raises(openai.AuthenticationError) as caught:
        client.chat.completions.create(model="@cf/test/e401", messages=MESSAGES)
    assert _classify(caught.value, cloudflare).is_auth


def test_a_server_error_is_retryable(stack):
    _, client, cloudflare = stack
    with pytest.raises(openai.InternalServerError) as caught:
        client.chat.completions.create(model="@cf/test/e500", messages=MESSAGES)
    assert _classify(caught.value, cloudflare).retryable is True


def test_the_model_list_comes_from_the_override_with_the_token(stack):
    profile, _, cloudflare = stack
    assert profile.fetch_models(api_key=TOKEN_A) == MODELS
    (seen,) = [s for s in cloudflare.seen if s.method == "GET"]
    assert seen.path.endswith("/ai/v1/models")
    assert seen.authorization == f"Bearer {TOKEN_A}"
