# CLAUDE.md — Forge

Project context for any Claude Code session working in this repo. Read this first.

## What this is
**Forge** — a task-specialization distillation pipeline. Give it one expensive, high-volume LLM task + a teacher model; it manufactures a verified small specialist model that matches teacher quality within a committed tolerance at 10–100× lower cost/latency, runnable fully private. **Not an agent.** This project deliberately proves *model manufacture* (SFT / distillation / DPO / serving economics / eval rigor), the counterpart to an LLM-orchestration portfolio.

## The one rule that overrides everything: INDEPENDENCE
This is a **personal portfolio project** built during an internship. It must depend on **nothing internal** to any employer:
- Teacher model: **open-weight** (Llama-3.x / Qwen2.5 / DeepSeek) or a public API whose ToS permits distillation. **Never** an internal/company-hosted model.
- Base model: open, **permissive license**.
- Data: **public datasets only** (or synthetic from the open teacher on public seeds). **No proprietary / internal / internship-acquired data, ever.**
- Compute: personal or commodity cloud. Code: OSS only. **No imports from any internal codebase (Aroha/Prism etc.).**
- **Litmus test (apply constantly):** "If internal access were cut tomorrow, could a stranger clone this repo and rebuild the model end-to-end?" If not yes — remove the dependency. See `docs/adr/0003`.

## Git / identity
- This is a **personal** repo on the **`Jatin-IITB`** GitHub account. Commits use the personal identity **`Jatin Gupta <22B3967@iitb.ac.in>`**, set via **local** repo config (never touch global, never the Paytm/work identity). Verify `git config user.email` returns `22B3967@iitb.ac.in` before committing.
- Auth is via the **`gh` CLI over HTTPS** (gh is the git credential helper). Remote: `https://github.com/Jatin-IITB/forge.git`.
- Never commit: proprietary/internal data, secrets/`.env`, large model weights or generated datasets (see `.gitignore`).

## Read in this order
1. `docs/DESIGN.md` — first-principles system design, novelty calibration, field comparison.
2. `docs/SUCCESS.md` — the six product gates + maturity rubric + what counts as failure.
3. `docs/ACTION_PLAN.md` — phased plan with exit gates, budget, risks.
4. `docs/adr/` — decision records (0001 eval-first, 0002 verification gate, 0003 independence).

## Current status
**Phase 0** (not started in code). Flagship task (recommended): **on-device PII detection & redaction** — public data, crisply measurable, privacy meaning is airtight. Alternatives: public-document extraction; an under-served-Indian-language specialist.

Phase 0 = lock the `TaskContract`, build & FREEZE a human-verified public gold set, pick open teacher + open base, clear licenses, pass the independence litmus test. **No modeling happens until Phase 0's gate passes (eval-first, `adr/0001`).**

## Working principles (Aroha doc discipline — portable practice)
- Eval-first: the frozen gold set + parity gate exist before the model.
- Verification-gated data: never train on the teacher's unverified output.
- Error-driven loop: spend teacher tokens where the student is wrong.
- Economics ($/1k, p95) are gates, not footnotes.
- Honest writeups: separate "genuinely novel" from "solid engineering"; compare to the field; state weaknesses.
- Reproducible: one command (`make forge`) rebuilds the asset; seeds fixed; versions pinned.
