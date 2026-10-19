# Cloudflare Workers AI for Hermes

A standalone model-provider plugin that adds [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
(OpenAI-compatible chat completions, `@cf/...` models) to [Hermes](https://github.com/NousResearch/hermes-agent).

> **Status: v0.1, unreleased, tested only against a throwaway Hermes home.** Not endorsed by Nous Research
> or Cloudflare. Licensed MIT (see LICENSE), matching hermes-agent. Pinned/tested against hermes-agent `fdec926ef5`.

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

Use a token scoped to **Workers AI only**. The token is sent as a bearer credential to whatever URL the
profile resolves to, so do not reuse a broad deploy/ops token here.

Install by placing this directory at `$HERMES_HOME/plugins/model-providers/cloudflare-workers-ai/`
(or `hermes plugins install <repo-url>`), then select the provider `cloudflare-workers-ai`
(alias `cf-workers-ai`).

## Behaviour worth knowing

- **Per-profile.** Each Hermes profile uses its own account, token and proxy, even when one process serves
  several profiles. A fixed account-free URL is registered with Hermes; the real one is applied when the
  client is built.
- **`CLOUDFLARE_BASE_URL` is read from the profile's `.env` only**, never from the shell or process
  environment: a variable exported in a shell (or inherited from a parent process) would otherwise
  redirect the token of every profile. The URL must be `https://` (plain `http://` only for `localhost`,
  `127.0.0.1` or `::1`) and carry no credentials, query string or fragment. The same checks apply to a
  base URL handed in by Hermes (`model.base_url`, a stored credential). A rejected URL sends nothing
  and the error never repeats it.
- **Capacity errors don't burn your credential pool.** Workers AI answers `429` / code `3040`
  ("Capacity temporarily exceeded") when the whole *account* is under contention. The plugin classifies
  it as `overloaded` (back off and retry) instead of Hermes's default `rate_limit`, which would rotate to
  the next pooled key even though every key shares the same limit. Other 429s keep Hermes's default.
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
python -m pip install -r requirements-test.txt
HERMES_AGENT_DIR=/path/to/hermes-agent python -m pytest tests -q
```

`HERMES_AGENT_DIR` is a checkout of [hermes-agent](https://github.com/NousResearch/hermes-agent) (Hermes
itself is not pip-installed). Three layers, all run by the one command:

- **Unit / contract tests** load the plugin through Hermes's real discovery inside throwaway Hermes homes
  and assert the HTTP boundary (which host, which path, which `Authorization`) across several profiles in
  one process, plus the URL-safety rules and the capacity-429 classifier.
- **End-to-end tests** (`tests/test_e2e_fake_cloudflare.py`) run the real OpenAI SDK over a real socket to
  a fake Cloudflare on `127.0.0.1` (`tests/fakecf.py`): streaming, `401`/`403`/`429`/`500`, `Retry-After`,
  and the verdict Hermes's own classifier reaches for each. No credentials, no network. They reach the
  *override* path only; the real `api.cloudflare.com` URL is deliberately not overridable, because a knob
  that redirected it would be a way to redirect the token. That one hop is covered by the unit tests.
- **CI** (`.github/workflows/test.yml`) runs the suite on every push and PR, and weekly, against two Hermes
  revisions: the **pin** (`fdec926ef5`, gating) and **main** (non-blocking early warning that Hermes
  changed something the plugin relies on - the tests use a few Hermes private APIs).

The suite **blocks** `hermes_bootstrap`: importing it (directly or via `agent.process_bootstrap`) outside
the Hermes launcher runs Hermes's launch preparation, which rebuilds the checkout's UI/desktop assets and
syncs dependencies over the network. The plugin itself only uses Hermes's client helpers when
`hermes_bootstrap` is already loaded.
