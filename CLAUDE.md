# CLAUDE.md — Forge

Project context for any Claude Code session working in this repo. Read this first.

## What this is
**Forge** — a task-specialization distillation pipeline. Give it one expensive, high-volume LLM task + a teacher model; it manufactures a verified small specialist model that matches teacher quality within a committed tolerance at 10–100× lower cost/latency, runnable fully private. **Not an agent.** This project deliberately proves *model manufacture* (SFT / distillation / DPO / serving economics / eval rigor), the counterpart to an LLM-orchestration portfolio.

## The one rule that overrides everything: INDEPENDENCE
This is a **personal portfolio project**, built independently. It must depend on **nothing private** — no employer's systems, data, or code:
- Teacher model: **open-weight** (Llama-3.x / Qwen2.5 / DeepSeek) or a public API whose ToS permits distillation. **Never** a privately hosted model behind a corporate account.
- Base model: open, **permissive license**.
- Data: **public datasets only** (or synthetic from the open teacher on public seeds). **No proprietary or otherwise private data, ever.**
- Compute: personal or commodity cloud. Code: OSS only. **No imports from any private codebase.**
- **Litmus test (apply constantly):** "If every private credential were revoked tomorrow, could a stranger clone this repo and rebuild the model end-to-end?" If not yes — remove the dependency. See `docs/adr/0003`.

## Git / identity
- This is a **personal** repo on the **`Jatin-IITB`** GitHub account. Commits use the personal identity **`Jatin Gupta <22B3967@iitb.ac.in>`**, set via **local** repo config (never touch global, never a work identity). Verify `git config user.email` returns `22B3967@iitb.ac.in` before committing.
- Auth is via the **`gh` CLI over HTTPS** (gh is the git credential helper). Remote: `https://github.com/Jatin-IITB/forge.git`.
- Never commit: proprietary or private data, secrets/`.env`, large model weights or generated datasets (see `.gitignore`).

## Read in this order
1. `docs/DESIGN.md` — first-principles system design, novelty calibration, field comparison.
2. `docs/SUCCESS.md` — the six product gates + maturity rubric + what counts as failure.
3. `docs/NORTH_STAR.md` — the claim ledger (public claims → repo evidence → status) + execution arcs.
4. `docs/ACTION_PLAN.md` — phased plan with exit gates, budget, risks.
5. `docs/adr/` — decision records (0001 eval-first … 0010 teacher scale-up).

## Current status
**Phase 4 — error-driven loop, in progress.** Task locked: **on-device PII detection & redaction**
(19 types, 9 high-severity). Frozen gold set: 574 records (385 test / 189 dev, Faker seed 42).
Student: Qwen2.5-1.5B-Instruct + LoRA (MPS fp16). Development teacher: Qwen3-8B (Ollama);
final teacher: GPT-OSS-120B via Cerebras free tier (`adr/0010`, key = `CEREBRAS_API_KEY`).
run_001 F1 = 0.52 → error analysis → construction-verified augmentation (`adr/0009`, seed 1337)
→ run_002 training. The claim ledger in `docs/NORTH_STAR.md` tracks what's proven vs pending.

## Working principles (documentation discipline — portable practice)
- Eval-first: the frozen gold set + parity gate exist before the model.
- Verification-gated data: never train on the teacher's unverified output.
- Error-driven loop: spend teacher tokens where the student is wrong.
- Economics ($/1k, p95) are gates, not footnotes.
- Honest writeups: separate "genuinely novel" from "solid engineering"; compare to the field; state weaknesses.
- Reproducible: one command (`make forge`) rebuilds the asset; seeds fixed; versions pinned.
