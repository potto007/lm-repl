# Architecture Decision Records

This directory holds [MADR](https://adr.github.io/madr/)-format Architecture
Decision Records for **prehend** (the patched RLM/SRLM orchestrator with an
experience-memory layer; renamed `lm-repl` -> `mnemex` per ADR-0006, then
`mnemex` -> `prehend` per ADR-0007 - older ADR titles keep the name they were
accepted under). Each ADR captures one
hard-won, non-obvious choice so it is found once and reversed only with evidence. New decisions get the next number; accepted ADRs are immutable (to
change course, write a new ADR and mark the old one `superseded by ADR-NNNN`).

Cross-repo references are written `<repo> ADR-NNNN`. The cross-repo master index
lives in [rlm-trainer `docs/decisions/README.md`](https://github.com/ClearBridgeRIP/rlm-trainer/tree/main/docs/decisions).

| # | Title | Status |
|---|-------|--------|
| [0001](0001-lm-repl-as-patched-fork-of-rlms.md) | lm-repl as a patched fork of `rlms` | accepted |
| [0002](0002-hard-per-generation-decode-token-ceiling.md) | Hard per-generation decode-token ceiling (`max_decode_tokens`) | accepted |
| [0003](0003-runaway-generation-guards.md) | Runaway-generation guards (soft-budget, subcall caps, contention-retry) | accepted |
| [0004](0004-reasoning-loop-repeat-guard.md) | Reasoning-loop repeat-guard + escalation (4th runaway-generation guard) | accepted |
| [0005](0005-mnemex-experience-memory-layer.md) | Adopt FinAcumen's FM as lm-repl's experience-memory layer (mnemex) | accepted |
| [0006](0006-rename-lm-repl-to-mnemex.md) | Full rename `lm-repl` -> `mnemex` (package, PyPI, repo) | superseded by [0007](0007-rename-mnemex-to-prehend.md) |
| [0007](0007-rename-mnemex-to-prehend.md) | Full rename `mnemex` -> `prehend` (package, PyPI, repo) | accepted |
| [0008](0008-high-level-harness-api.md) | High-level `Harness` API (Tier A/B/C; runtime detection) | accepted |
| [0009](0009-subcall-input-context-guard.md) | Sub-call input-size context guard (reject-with-hint; 1st input-axis guard) | accepted |
| [0010](0010-auto-chunk-enforcement-for-oversized-subcalls.md) | Auto-chunk enforcement for oversized sub-calls (`context=` map-reduce) | accepted |
| [0011](0011-contrastive-failure-memory-channel.md) | Contrastive failure memory channel (negative guard rules from wrong solves) | accepted |
| [0012](0012-pool-aware-subcall-budget-under-kv-unified.md) | Pool-aware sub-call budget: divide the shared kv-unified pool across concurrent sub-calls | accepted |
| [0013](0013-dual-instance-weight-shared-inference.md) | Dual-instance weight-shared inference: split orchestrator and sub-calls onto two processes sharing one weights copy | superseded by [0014](0014-single-process-dual-context-inference.md) |
| [0014](0014-single-process-dual-context-inference.md) | Single-process dual-context inference: one `llama_model` backing two `llama_context` (private KV each) | accepted |
| [0015](0015-inference-engine-evaluation-vllm-sglang.md) | Inference-engine evaluation: spike vLLM and SGLang as single-engine replacements for the dual-context fork | proposed |
| [0016](0016-sglang-as-served-inference.md) | SGLang as the served inference: retire the dual-context llama.cpp fork (GATE #1 pass; GATE #2 accuracy A/B pending) | proposed |
| [0017](0017-data-first-subcall-layout.md) | Data-first sub-call layout (context before instruction) to fix prefix misalignment / re-prefill | accepted |
| [0018](0018-extraction-map-for-multihop-chaining.md) | Query-independent extraction MAP for multihop chaining | accepted |
| [0019](0019-fp8-e4m3-kv-cache-for-gemma4.md) | fp8_e4m3 (not fp8_e5m2) for the gemma4 v13 KV cache | accepted |
| [0020](0020-entry-id-includes-provenance.md) | Experience id keys on (question, provenance) so failure guards and success recipes coexist (amends 0011) | accepted |
| [0021](0021-vllm-as-served-inference.md) | vLLM 0.23.0 as the served inference for v13 (TRITON_ATTN, fp8_e4m3 KV, max-model-len 65536); supersedes 0016 | accepted |
| [0022](0022-cold-baseline-via-frozen-retrieval-and-persistent-bank.md) | Cold baseline = frozen retrieval (not an empty bank); the memory bank is persistent and versioned | accepted |
| [0023](0023-per-scope-subcall-attribution.md) | Per-root subcall attribution via a scope-bound callback, not broadcast | accepted |

`0000-template.md` is the MADR template for new records.
