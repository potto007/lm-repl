# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Machine-wide rules (the shared GPU, prod vLLM server, systemd units, monitoring rail, git/code conventions) live in ~/.claude/CLAUDE.md and are loaded in every session. This file holds only what is specific to prehend.

## What this is

prehend (originally lm-repl; import path `prehend`, repo `potto007/prehend`, on-disk dir `~/src/prehend`) is a language-model harness that learns. Context is offloaded into a variable inside a REPL environment, and the model writes programs that slice, search, and recursively query that context instead of attending over it directly; an experience-memory layer (`prehend/memory/`) distills and retrieves past solves so each run improves on the last. See ADR-0007 for the rename to prehend and ADR-0005 for the memory layer.

This venv uses `uv` (there is NO `pip` binary in `.venv/bin`; use `~/.local/bin/uv`).

**CRITICAL**: This instruction file must not be modified without gaining explicit user permission first. You may edit any other files in the project, including README.md, but this file must remain unchanged unless user grants permission.

## llama-server (the harness's served Gnosis model): diagnosing endpoint timeouts

(Production inference now runs on vLLM on :8080 per ADR-0021; this section documents the llama-server router-mode path, retained as the rollback reference.) The WSL2 llama-server on localhost:8080 (gemma-4 GGUFs) serves **Gnosis** - our custom model FAMILY trained to operate in the **RLM pattern** inside the REPL the prehend harness provides; the harness drives one as its **inference engine/orchestrator**. Each Gnosis model is a domain specialty named `Gnosis-<Specialty>-<version>`; current = **Gnosis-MedPolicy-13** (`gemma-4-12b-it-sft-kb-v13-sft`). Use bare "Gnosis" only for the family, never bare "v13", and **NEVER call a Gnosis model "teacher"/"student"** (those apply ONLY to training-trajectory generation in rlm-trainer). [HF packaging convention in auto-memory `reference_gnosis-model-name`: org `gnosis-lm`, one repo per specialty, version as a tag.] When a run hitting this server times out (`ConnectTimeout`/`APITimeoutError`) but `curl`/another process to the "same" endpoint works and the server log shows `all slots are idle`:

**FIRST, check the actual socket destination: `ss -tnp | grep <pid>`.** A real case burned hours: the run was `SYN-SENT` to `172.19.144.1:1234` (the RETIRED Windows LM Studio endpoint), not `localhost:8080`, because `make_kb_realism.py` lacked `load_dotenv()` and `base_url` fell back to its hardcoded dead default. "Server idle + curl fast + run times out" = the run isn't talking to that server. Confirm the wire before tuning the server. (Scripts that hit this server MUST `load_dotenv()` like `generate.py` does, or be passed the base-url explicitly; rlm-trainer's training-data-gen scripts read it from `TEACHER_BASE_URL`.)

Real server/client notes:
- **Router mode WORKS and is the current setup (NOT a plain single-model server).** Launch: `llama-server --models-preset ~/src/local-ai/models/rlm-models.ini --models-max 1 --no-models-autoload --metrics` (the `.ini` + convenience scripts live in `~/src/local-ai`, repo `ClearBridgeRIP/local-ai`; prefer `~/src/local-ai/scripts/llama-server.sh start|load|stop`). Three non-obvious gotchas, all settled empirically 2026-06-01 - the router kept loading/OOMing the 26B nobody requested until all three were fixed:
  1. **No content before the first `.ini` section** (`[*]` or a `[model]`) - not even a `version = 1` header or a comment. Anything before the first section is vacuumed into a phantom `default` model that the router auto-loads on startup and keeps warm with a ~10s keepalive, evicting/OOMing whatever you actually request (llama.cpp issue #22364).
  2. **`--no-models-autoload`** for pure on-demand: without it, `--models-autoload` (the default) auto-loads the FIRST preset section at startup. With it, nothing loads until requested - BUT a chat to an unloaded model then 400s ("model is not loaded"); load explicitly first via `POST /models/load {"model":"<id>"}` (endpoint is `/models/load`, not `/load`).
  3. **Use cache-reuse, NOT `--cache-ram 0`.** The RLM orchestrator reuses a long system-prompt PREFIX every iteration, so prefix-cache reuse is a real ~5-10x win (verified: request 2 reused 3748/3760 prefix tokens). `cache-reuse=256` needs `swa-full=true` to fire on gemma-4's sliding-window attention (#22288). swa-full forces FULL-size SWA KV (KV scales with TOTAL ctx-size, not parallel), which OOMs at f16 ctx 65536 on the 32GB 5090 -> fix with **`q8_0` KV** (symmetric K+V, halves KV; needs flash-attn on; the prior failure was asymmetric V-only). Result: dpo runs `ctx-size 65536 / parallel 2` (two 32768-tok slots, each holds the ~20K-tok multihop REDUCE) at ~28GB with headroom. Set `ctx-size`/`parallel` PER MODEL in named sections (dpo 65536/2, the current v13 sft model `gemma-4-12b-it-sft-kb-v13-sft` 98304/4, 26B 16384/1); client sets `MAPREDUCE_CONCURRENCY` to match the slot count. **`--kv-unified` (set in the systemd unit) makes `ctx-size` ONE pool SHARED across all `parallel` slots, NOT a private window per slot** - so a single task's concurrent map-reduce sub-calls contend for the same pool. The client guard must therefore budget each sub-call at `ctx-size / slots`, not the whole window (prehend ADR-0012, `per_call_subcall_budget`); budgeting each at the full pool exhausts the shared cache (`failed to find free space in the KV cache` -> `Context size has been exceeded` in bursts of `parallel` task ids).
- **httpx pool:** OpenAI SDK default (`max_connections=1000`, keep-alive on) is good; `generate.py` uses it. Don't set `max_keepalive_connections=0` (churn -> PoolTimeout; openai-python #2539/#763).
- **Orphans:** SIGTERM won't kill a python client blocked in an httpx timeout; before launch `ps -eo cmd | grep '[.]venv/bin/python'` must be empty (SIGKILL + confirm; beware grep self-match).

Ops: kill the server by explicit PID (NEVER `pkill -f llama-server` - self-match, exit 144); confirm port 8080 has 0 listeners + VRAM back to idle (~2GB) before relaunch; relaunch needs `LD_LIBRARY_PATH=/usr/local/cuda-13/lib64`. Validate config changes with a SUSTAINED run, not a burst.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes - gives risk-scored analysis |
| `get_review_context` | Need source snippets for review - token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
