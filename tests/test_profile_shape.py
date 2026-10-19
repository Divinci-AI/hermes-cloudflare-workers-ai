"""Small facts about the registered profile that Hermes relies on and that are easy to break silently."""

from __future__ import annotations

from support import ACCOUNT_A, PROVIDER, TOKEN_A, in_home
from test_routing import call


def _profile(make_home):
    from providers import get_provider_profile, list_providers

    with in_home(make_home({})):
        list_providers()
        return get_provider_profile(PROVIDER)


def test_the_doctor_probe_stays_off_because_it_would_hit_the_placeholder_url(make_home):
    assert _profile(make_home).supports_health_check is False


def test_there_is_a_fallback_model_and_it_is_a_usable_slug(make_home):
    models = _profile(make_home).fallback_models
    assert models and all(m.startswith("@cf/") for m in models)


def test_the_classifier_hook_is_registered(make_home):
    assert callable(_profile(make_home).classify_api_error)


def test_quotes_around_env_values_are_stripped(make_home):
    """A hand-edited .env often quotes values; the account id must still reach the URL unquoted."""
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A, "CLOUDFLARE_ACCOUNT_ID": f'"{ACCOUNT_A}"'})
    assert call(home)[0][0].url.path == f"/client/v4/accounts/{ACCOUNT_A}/ai/v1/chat/completions"


def test_quotes_are_stripped_from_a_process_env_value_too(make_home, monkeypatch):
    """Hermes's dotenv parser unquotes .env itself, so the plugin's own strip only matters for the
    process-environment fallback (an account id exported with literal quotes, e.g. from a CI secret)."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", f"'{ACCOUNT_A}'")
    home = make_home({"CLOUDFLARE_API_TOKEN": TOKEN_A})
    assert call(home)[0][0].url.path == f"/client/v4/accounts/{ACCOUNT_A}/ai/v1/chat/completions"
