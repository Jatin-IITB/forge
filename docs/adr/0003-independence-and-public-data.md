# ADR 0003 — Zero internal dependency: public data + open weights + own compute

**Status:** Accepted
**Date:** 2026-06-10

## Context
This project is built during an internship. **All access to internal systems (Aroha/Prism code, internal hosted frontier models, any proprietary/company data) ends with the internship.** A portfolio asset that silently depends on any of those becomes unbuildable, unverifiable, and unshippable the day access is revoked. It would also be unethical to redistribute internal data or model outputs.

## Decision
Forge depends on **nothing internal**, by hard rule:

1. **Teacher model** — an **open-weight** model (e.g. Llama-3.x, Qwen2.5, DeepSeek), or a public API *whose terms explicitly permit distillation and redistribution*. **Never** an internal/company-hosted model you will lose access to.
2. **Base (student) model** — an open model under a **permissive license** that allows fine-tuning and release.
3. **Data** — **public datasets** (with distillation-compatible licenses) or synthetic data generated *by the open teacher from public seeds*. **No internal, proprietary, or internship-acquired data.**
4. **Compute** — personal hardware or commodity cloud (Colab/spot GPU). No internal clusters.
5. **Code** — standard OSS only (`transformers`, PEFT, TRL, vLLM, llama.cpp, Outlines, etc.). No imports from Aroha/Prism. The *only* thing borrowed from Aroha is documentation discipline, which is portable practice, not a dependency.

## Consequences
- (+) The asset survives the internship intact: reproducible and shareable by anyone, forever.
- (+) Cleanly open-source-able and legally safe to publish.
- (+) Forces the flagship task toward problems with strong **public** datasets (PII, public documents) — which also happen to be the most *meaningful* and demoable.
- (−) Loses the "trained on real proprietary fintech data" angle. Accepted: permanence + reproducibility + shareability outweigh a one-time data-realism story. The realism gap is closed by choosing tasks with good public corpora.

## Litmus test (applied before Phase 0 exits)
> "If my internship ended tomorrow and my access was cut, could a stranger clone this repo and rebuild the model end-to-end?" If the answer is anything but **yes**, the dependency is removed before proceeding.
