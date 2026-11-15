# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/afrzh11/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/afrzh11/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

One server (Node 22, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API over ten simulated free-tier
model providers, in JSON and in server-sent events, and registers the catalog
it read from them.

## Run it

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog as one JSON document and exit 0
```

## How it routes

Every request names model `auto`; the server picks the model:

- **Best first.** The candidates are the free models whose capabilities meet
  the request (tools, vision, JSON mode, context length), ranked by the
  quality table's `rating_milli`, ties by provider id then model id.
- **JSON mode stays on a wire that can carry it.** A JSON-mode request is
  only routed to an OpenAI or Gemini provider: the Anthropic wire has no
  `response_format`, so its echo would say JSON mode was absent.
- **Failover.** A 402, 429, 5xx or unreachable provider costs one call and
  the request is answered by the next best model that meets it. A provider
  that has refused twice in a row is tried last: its free-tier budget is
  most likely spent. Each candidate is tried at most once per request, and
  the first success ends it — the trial counts exactly one successful
  provider call per request.
- **Faithful wires.** The system prompt, user text (images included), tools,
  stop sequences, `max_tokens` and JSON mode are carried across the OpenAI,
  Anthropic and Gemini wires exactly, because the provider echoes what it
  parsed and the client compares that echo with what it sent. Stop reasons
  map end_turn to stop, max_tokens to length, tool_use to tool_calls and
  refusal to content_filter; usage counts are passed through as the wire
  reports them, including the usage chunk on OpenAI streams.
- **Deterministic.** No wall clock, no unseeded randomness: a replay leaves
  the same state export, byte for byte.

## Layout

```
main.ts            the server: brief, catalog, routes, register, poll, --list
lib/brief.ts       the brief's three documents, the catalog probe, ranking
lib/translate.ts   the three wires, both ways, JSON and SSE
LICENSE            MIT
```

## Practice scores (sealed env + scorer, this machine)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

20 of 20 requests valid on every trial; the `--list` catalog matches the
sealed reference on all ten providers; running trial 0 twice exports
byte-identical state. Quality 99 is the ceiling: the best model (1475) has a
10–12 call budget for the 17 requests that want it, so some must fall to
1470.
