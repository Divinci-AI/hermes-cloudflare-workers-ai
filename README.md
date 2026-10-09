# Cloudflare Workers AI for Hermes

A standalone model-provider plugin that adds [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
(OpenAI-compatible chat completions, `@cf/...` models) to [Hermes](https://github.com/NousResearch/hermes-agent).

> **Status: v0.1, unreleased, tested only against a throwaway Hermes home.** Not endorsed by Nous Research
> or Cloudflare. Licence not yet chosen. Pinned/tested against hermes-agent `fdec926ef5`.

## Why a plugin

Hermes's contribution policy keeps third-party vendor integrations out of the core tree, so this ships as a
user plugin. Workers AI is account-scoped (`.../accounts/{account_id}/ai/v1`), which core's provider registry
cannot express, so the plugin resolves the URL per call (see `endpoint.py`).

## Setup

Put these in the Hermes profile's `.env`:

```
CLOUDFLARE_API_TOKEN=...      # needs Workers AI read/run; CLOUDFLARE_API_KEY is also accepted
CLOUDFLARE_ACCOUNT_ID=...     # Cloudflare dashboard -> Workers AI
# optional: CLOUDFLARE_BASE_URL=https://your-gateway.example/ai/v1   (proxy / AI Gateway)
```

Install by placing this directory at `$HERMES_HOME/plugins/model-providers/cloudflare-workers-ai/`
(or `hermes plugins install <repo-url>`), then select the provider `cloudflare-workers-ai`
(alias `cf-workers-ai`).

## Behaviour worth knowing

- **Per-profile.** Each Hermes profile uses its own account, token and proxy, even when one process serves
  several profiles. A fixed account-free URL is registered with Hermes; the real one is applied when the
  client is built.
- **Account id changes apply without a restart.**
- **Missing or malformed `CLOUDFLARE_ACCOUNT_ID`** fails with a clear error in the conversation and sends
  nothing over the network.
- **`.env` beats the credential pool.** Hermes reads `.env` before keys stored with `hermes auth add`, so a
  stale `CLOUDFLARE_API_KEY` in `.env` silently wins over a newer stored key.
- **The model list** comes from the account's `/ai/models/search` (paged), filtered to chat-capable tasks.
  With `CLOUDFLARE_BASE_URL` set, that proxy's `/models` is used instead.
- **Known limits (v0.1):** `hermes doctor`'s provider probe is disabled (it would hit the placeholder URL);
  `hermes model` does not prompt for the account id; a custom CA bundle configured in Hermes is not applied
  to this provider's client; the model-list cache is not keyed on the account id.

## Tests

```
HERMES_AGENT_DIR=~/.hermes/hermes-agent PYTHONPATH=.testdeps \
  ~/.hermes/hermes-agent/venv/bin/python -m pytest -c tests/pytest.ini tests
```

The suite runs inside throwaway Hermes homes and **blocks** `hermes_bootstrap`: importing it (directly or via
`agent.process_bootstrap`) outside the Hermes launcher runs Hermes's launch preparation, which rebuilds the
checkout's UI/desktop assets and syncs dependencies over the network. The plugin itself only uses Hermes's
client helpers when `hermes_bootstrap` is already loaded.
