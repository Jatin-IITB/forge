#!/usr/bin/env python3
"""Score gate G6 — out-of-domain handling and adversarial robustness.

Reports the two axes separately and never averages them, because they fail in
opposite directions and a single score hides both failure modes:

- A model that returns empty for everything scores 100% on out-of-domain and 0%
  on adversarial. Averaged, it looks mediocre rather than dangerous.
- A model that over-detects scores the reverse.

**G6 passes only if both rates clear the contract threshold.**

Usage:
    python scripts/run_ood_eval.py data/ood_probe.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct --adapter checkpoints/run_002/final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from forge.contracts import load_contract
from forge.inference import build_messages, parse_response
from forge.ood import detect_out_of_domain
from forge.validators import find_high_severity, merge_with_model


def load_probes(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spans_for(text: str, raw: str, use_validators: bool) -> list:
    record, _ = parse_response("probe", text, raw, split="test")
    spans = list(record.spans) if record else []
    if use_validators:
        spans = merge_with_model(spans, find_high_severity(text))
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description="Score gate G6 (OOD + adversarial).")
    ap.add_argument("probes", type=Path, help="Probe JSONL from build_ood_probe.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument(
        "--validators", action="store_true",
        help="Include the ADR 0012 validator layer (scores the system, not the model)",
    )
    ap.add_argument(
        "--ood-gate", action="store_true",
        help=(
            "Apply the contract's out-of-domain gate (forge/ood.py) before the model. "
            "Refused inputs return an empty span list and never reach the model or "
            "the validators"
        ),
    )
    ap.add_argument("--contract", type=Path, default=Path("contracts/pii_redaction_v2.yaml"))
    ap.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    contract = load_contract(args.contract)
    threshold = contract.gates.ood_refusal_min

    generate = _make_generator(args)

    results = []
    for i, probe in enumerate(probes, 1):
        t0 = time.monotonic()
        # The gate runs BEFORE the model, which is the whole point: a refused
        # document costs no tokens and gives neither the model nor the
        # validators a chance to invent a span. Skipping the call is also where
        # the economics win comes from — OOD inputs otherwise generate to the
        # token cap, 13.3 s against 2.6 s in-domain.
        verdict = detect_out_of_domain(probe["text"]) if args.ood_gate else None
        if verdict is not None and verdict.is_ood:
            latency = time.monotonic() - t0
            spans = []
            refused = verdict.reason
        else:
            raw = generate(probe["text"])
            latency = time.monotonic() - t0
            spans = _spans_for(probe["text"], raw, args.validators)
            refused = None

        if probe["category"] == "out_of_domain":
            # Correct = invented nothing.
            passed = len(spans) == 0
            detail = f"{len(spans)} spans" if spans else "clean"
        else:
            found = " ".join(s.text for s in spans)
            missing = [m for m in probe["must_detect"] if m not in found]
            passed = not missing
            detail = "all detected" if passed else f"missed {missing}"

        results.append(
            {
                "id": probe["id"],
                "category": probe["category"],
                "passed": passed,
                "detail": detail,
                "n_spans": len(spans),
                "latency_s": round(latency, 3),
                "refused_by_gate": refused,
            }
        )
        mark = "ok  " if passed else "FAIL"
        print(f"  [{i}/{len(probes)}] {mark} {probe['id']:<38} {detail}")

    ood = [r for r in results if r["category"] == "out_of_domain"]
    adv = [r for r in results if r["category"] == "adversarial"]
    ood_rate = sum(r["passed"] for r in ood) / len(ood) if ood else 0.0
    adv_rate = sum(r["passed"] for r in adv) / len(adv) if adv else 0.0
    gate_pass = ood_rate >= threshold and adv_rate >= threshold

    print()
    print("=" * 60)
    print("  GATE G6 — out-of-domain handling & adversarial robustness")
    print("=" * 60)
    print(f"  out-of-domain (no invented spans) : {ood_rate:.4f}  ({sum(r['passed'] for r in ood)}/{len(ood)})")
    print(f"  adversarial   (still detects PII) : {adv_rate:.4f}  ({sum(r['passed'] for r in adv)}/{len(adv)})")
    print(f"  threshold (both must clear)       : {threshold:.4f}")
    print(f"  verdict                           : {'PASS' if gate_pass else 'FAIL'}")
    print()
    print("  Rates are reported separately by design: a model that returns")
    print("  empty for every input would score 1.0 on the first line and 0.0")
    print("  on the second, and averaging would disguise that.")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print()
        print(f"  {len(failures)} failing probes:")
        for r in failures:
            print(f"    {r['category']:<15} {r['id']:<38} {r['detail']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "ood_pass_rate": ood_rate,
                    "adversarial_pass_rate": adv_rate,
                    "threshold": threshold,
                    "gate_pass": gate_pass,
                    "model": args.model,
                    "adapter": str(args.adapter) if args.adapter else None,
                    "validators": args.validators,
                    "ood_gate": args.ood_gate,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  wrote {args.output}")

    return 0 if gate_pass else 1


def _make_generator(args):
    """Return a text -> raw-response function for either local or API inference."""
    if args.adapter:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        use_mps = torch.backends.mps.is_available()
        device = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if use_mps else torch.bfloat16
        print(f"loading {args.model} + {args.adapter} on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
        model.eval()

        def generate(text: str) -> str:
            messages = build_messages(text)
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = (
                encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)

        return generate

    import os

    from openai import OpenAI

    key = os.environ.get(args.api_key_env, "not-needed") if args.api_key_env else "not-needed"
    if args.api_key_env and key == "not-needed":
        print(f"{args.api_key_env} is not set", file=sys.stderr)
        raise SystemExit(1)
    client = OpenAI(base_url=args.base_url, api_key=key)
    interval = 60.0 / args.rpm if args.rpm else 0.0
    last = [0.0]

    def generate(text: str) -> str:
        if interval:
            wait = interval - (time.monotonic() - last[0])
            if wait > 0:
                time.sleep(wait)
        last[0] = time.monotonic()
        resp = client.chat.completions.create(
            model=args.model,
            messages=build_messages(text),
            max_tokens=args.max_tokens,
            temperature=0.0,
            extra_body=(
                {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
            ),
        )
        return resp.choices[0].message.content or ""

    return generate


if __name__ == "__main__":
    raise SystemExit(main())
