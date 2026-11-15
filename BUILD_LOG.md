# Build log — free-model-aggregator-01

## What was built

One Node 22 server (stdlib only), written from scratch in this session:

- `main.ts` — the server: reads the brief, probes the ten providers, builds
  the one catalog, registers `{protocol_version: 1, port, catalog}` at
  `$MAKEFUN_ENV_URL/register`, serves `POST /v1/chat/completions`,
  `POST /v1/messages` and `GET /v1/models`, polls `GET /work` every 100 ms
  and exits 0 when it says done. `--list` prints the catalog and exits 0.
- `lib/brief.ts` — the brief's three JSON documents (found by `schema`),
  the catalog probe in each provider's own wire, the catalog builder, and
  the candidate ranking (free + meets requirements, rating desc, provider
  then model id for ties; JSON-mode requests exclude the anthropic wire).
- `lib/translate.ts` — the three wires both ways, JSON and SSE: request
  translation (system, user text, images, tools, stops, max_tokens, JSON
  mode, temperature, tool_choice), answer translation (text, usage, stop
  reasons), and the four cross-wire stream converters plus passthrough.

## Key facts learned from the kit (read before writing code)

- The provider answers `ANSWER ECHO<json>`: the client compares the echo
  (system/user sha, tool names, max_tokens, stops, image sha, json_mode)
  with what it sent, so every translated field must arrive exactly.
- Exactly one successful provider call per request counts; a second success
  voids it. Failures are 402 on pollinations, 429 on the other nine, some
  with retry-after; budgets are 4–16 calls per provider per trial, drawn
  per trial, plus per-provider fail-call schedules and one outage.
- The usage check needs usage on OpenAI streams too, so the upstream call
  always sets `stream_options.include_usage: true` and the usage chunk is
  forwarded/emitted; on Anthropic streams only the output count is checked.
- The stop family the provider answers is always the "stop" family, so the
  entry reports `stop` / `end_turn` respectively.
- `--list` and the register body must be the same document; the catalog's
  base_url is the directory's path (`/p/<id>`), never joined to the env URL.

## Bugs found by the practice trials

1. First run scored 67 (13/20 valid): an Anthropic client body routed to an
   OpenAI-wire provider was passed through untranslated, so the provider
   echoed no system, no tools and no stops. Fixed in `upstreamFor`: the
   Anthropic body is rewritten with `anthropicToOpenAIReq` before the call.
   Trial 0 then scored 99.

## Practice results (sealed env + sealed scorer, run manually: env on
# 127.0.0.1:8080, bot with MAKEFUN_ENV_URL / MAKEFUN_PROMPT_FILE /
# MAKEFUN_TRIAL_SEED, SIGTERM the env to export state, then score.mjs)

- trial-0: **99** (quality 99, served 100, wire 100, failover 100, catalog
  100), 20/20 valid, 27 upstream calls.
- trial-1: **99** (99/100/100/100/100), 20/20 valid, 30 upstream calls.
- trial-2: **99** (99/100/100/100/100), 20/20 valid, 29 upstream calls.
- `--list` against a live trial-0 env matches the sealed reference catalog
  on all ten providers (free/paid sets, base_url, key_env, wire).
- Trial 0 run twice exports byte-identical `state.json` (replay check).
- Quality 99 is the ceiling: the 1475 model (openrouter) has a 10–12 call
  budget for the 17 requests that want it; the theoretical max earned is
  38238/38248 = 99.97, which floors to 99. Trial-0 earned 38233.

## Not verified

- The real Node 22 runtime: this machine has Node 24.22.1; the code uses
  only APIs present in Node 22 (fetch, AbortSignal.timeout, async
  generators, TextDecoder) and erasable TypeScript syntax, but it was not
  executed under Node 22.
- The scored instances are drawn after entries close (drand round
  2026-11-16T02:40Z); only the three practice instances were run.

## Bundle + check (parent, 2026-11-15 ~18:46 WIB)

Usage row sesi builder ada di muse.db (message `7f3c209c-91a2-440d-a640-4b5e2f0f99f6`,
8.103.927 in / 54.971 out), disimpan di `~/workspace/makefun/free-aggregator-usage.jsonl`
(di luar repo agar tree tetap bersih). Bundle: `~/workspace/makefun/free-aggregator-bundle.tar.gz`
(158.620 byte, sha256 `8a1d2aa11baf6be85f112b6aea03435de7b60d6dff798fb0d4fdb70cdcc8a67c`,
1 sesi). `makefun check`: **15 passed, 0 failed**; `thinking_reproduces` 860/1553 = 55,3%
(main.ts 159 authored lines), `thinking_priced` 1 record. **Belum di-push / belum
di-submit — menunggu persetujuan user.** Commit yang akan di-submit: `ea73344`.
