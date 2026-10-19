"""Cloudflare's account-level capacity 429 (code 3040) must back off, not rotate credentials.

Asserted two ways: the pure matcher, and the verdict Hermes's REAL ``classify_api_error`` returns for an
error shaped like the one the OpenAI SDK raises - which is what proves the hook is registered and
consulted, not merely correct in isolation.
"""

from __future__ import annotations

import importlib.util

import httpx
import openai
import pytest

from support import PROVIDER, REPO

_spec = importlib.util.spec_from_file_location("cfwai_classify", REPO / "classify.py")
classify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classify)

CAPACITY_BODY = {"errors": [{"code": 3040, "message": "AiError: Capacity temporarily exceeded, please try again"}],
                 "success": False}


def _match(status=429, code=None, message="", body=None):
    return classify.is_capacity_error(status_code=status, error_code=code, message=message, body=body)


# ---- the matcher -------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(body=CAPACITY_BODY),                                            # Cloudflare envelope
    dict(body={"error": {"code": 3040, "message": "x"}}),                # OpenAI-style envelope, int code
    dict(body={"error": {"code": "3040"}}),                              # string code
    dict(code=3040),                                                     # surfaced as error_code
    dict(code="3040"),
    dict(message="AiError: Capacity temporarily exceeded, please try again"),
    dict(message="CAPACITY TEMPORARILY EXCEEDED"),                       # case-insensitive
    dict(body='{"errors":[{"message":"Capacity temporarily exceeded"}]}'),  # body kept as a string
])
def test_the_capacity_shape_is_recognised(kwargs):
    assert _match(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(status=500, body=CAPACITY_BODY),                                # same text, wrong status
    dict(status=503, body=CAPACITY_BODY),
    dict(status=None, body=CAPACITY_BODY),
    dict(body={"errors": [{"code": 10000, "message": "Daily request limit exceeded"}]}),   # ordinary quota
    dict(body={"errors": [{"code": 2005, "message": "Rate limit exceeded"}]}),
    dict(message="request id 3040abc was throttled"),                    # 3040 inside another token
    dict(message="exceeded 13040 bytes"),
    dict(body={"errors": [{"code": 30400}]}),
    dict(code=30400),
    dict(),
])
def test_anything_else_is_declined(kwargs):
    assert not _match(**kwargs)


def test_the_hook_declines_by_returning_none_so_hermes_runs_its_own_pipeline():
    assert classify.classify_api_error(Exception("x"), status_code=429, error_code=None, message="slow down",
                                       body={"errors": [{"code": 2005}]}, model="m") is None
    assert classify.classify_api_error(Exception("x"), status_code=401, error_code=None, message="no",
                                       body={}, model="m") is None


def test_the_verdict_is_overloaded_retryable_and_never_rotates():
    verdict = classify.classify_api_error(Exception("x"), status_code=429, error_code=None,
                                          message="", body=CAPACITY_BODY, model="@cf/x/y")
    assert verdict == {"reason": "overloaded", "retryable": True, "should_rotate_credential": False}


# ---- through Hermes's real classifier ------------------------------------------------------------------

def _rate_limit_error(body, headers=None):
    request = httpx.Request("POST", "https://api.cloudflare.com/client/v4/accounts/a/ai/v1/chat/completions")
    response = httpx.Response(429, json=body, headers=headers or {}, request=request)
    return openai.RateLimitError(message=str(body), response=response, body=body)


def _classify(error, make_home):
    from agent.error_classifier import classify_api_error
    from providers import list_providers

    from support import in_home

    with in_home(make_home({})):
        list_providers()  # Hermes only consults a profile it has discovered in the bound home
        return classify_api_error(error, provider=PROVIDER, model="@cf/x/y",
                                  base_url="https://api.cloudflare.com/client/v4/ai/v1", api_key="k")


def test_hermes_classifies_the_capacity_429_as_overloaded_without_rotating(make_home):
    verdict = _classify(_rate_limit_error(CAPACITY_BODY, {"Retry-After": "7"}), make_home)
    assert verdict.reason.name == "overloaded"
    assert verdict.retryable is True
    assert verdict.should_rotate_credential is False


def test_an_ordinary_cloudflare_429_keeps_hermes_default_verdict(make_home):
    """Declining must leave the built-in behaviour exactly as it was - rate_limit, which rotates."""
    verdict = _classify(_rate_limit_error({"errors": [{"code": 2005, "message": "Rate limit exceeded"}]}), make_home)
    assert verdict.reason.name == "rate_limit"
    assert verdict.should_rotate_credential is True


def test_the_hook_is_what_changes_the_verdict(make_home, monkeypatch):
    """Without the hook Hermes would rotate: proves the test above is not passing for another reason."""
    from providers import get_provider_profile, list_providers

    from support import in_home

    with in_home(make_home({})):
        list_providers()
        monkeypatch.setattr(get_provider_profile(PROVIDER), "classify_api_error", None)
        from agent.error_classifier import classify_api_error
        verdict = classify_api_error(_rate_limit_error(CAPACITY_BODY), provider=PROVIDER, model="@cf/x/y",
                                     base_url="https://api.cloudflare.com/client/v4/ai/v1", api_key="k")
    assert verdict.reason.name == "rate_limit" and verdict.should_rotate_credential is True
