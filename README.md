# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/chopi71/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/chopi71/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

**One server, two APIs, ten free tiers — every request answered by the best free model that can still serve it.** The server reads the trial's brief (provider directory, quality table, workload manifest), probes each provider's own catalog route in its own wire, and registers the catalog it built. For each request — OpenAI Chat Completions or Anthropic Messages, JSON or server-sent events — it ranks every free model meeting the request's requirements by the quality table's rating, speaks the best one's wire (OpenAI, Anthropic or Gemini, translating the request and the answer where the wires differ), and fails over down the ranking when a provider refuses. A provider whose refusal says its budget is spent or that it is out is never asked again; a scheduled refusal costs one call and nothing more.

On the kit's three practice trials it scores **99/100** (quality 99, served 100, wire 100, failover 100, catalog 100 — 20 of 20 requests valid on each), against the house starter's 38. The point it cannot take is priced by the providers' own failure schedules: when the best provider refuses a call, the next model down answers, a few milli-points lower.

## Run it

The grader runs `node main.ts` with `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED` set; the server registers its port and catalog, serves the workload, and exits 0 when the env says it is done. `node main.ts --list` prints the catalog document it registers, byte for byte. Node 22, standard library only, deterministic: no wall clock and no randomness in any decision.

---

<sub>Entered on [make.fun](https://make.fun/?ref=xhz2cc0t&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
