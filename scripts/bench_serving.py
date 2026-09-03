#!/usr/bin/env python3
"""Serving benchmark — the timing harness behind gates G3 (cost) and G4 (latency).

Why this exists separately from `run_inference.py`: that script measures a
*correctness* run and reports per-request latency. Gate G3 is not a per-request
number. It is **machine-seconds per record**, and under concurrency those two
quantities diverge sharply — eight requests each taking 3 s but overlapping cost
the machine 0.4 s/record, not 3 s/record. Conflating them would silently inflate
or deflate the cost gate depending on which config was measured, so the sustained
figure is computed here from wall-clock over the whole set and carried explicitly.

The harness measures two configurations that pull against each other:

    latency config     concurrency 1  -> p50/p95 per request        (gate G4)
    throughput config  concurrency N  -> sustained s/record         (gate G3)

Both are always reported. Quoting the p95 from the batch-1 run next to the $/1k
from the batched run, as if one config produced both, is the specific dishonesty
this harness is built to prevent — so the config name is stamped into every
artifact and every derived number.

Backends
    openai        an OpenAI-compatible endpoint (llama-server, vLLM, Ollama)
    transformers  local HF model on MPS/CUDA/CPU, batch 1 — the incumbent baseline

Usage
    # baseline: the incumbent transformers-MPS fp16 path
    python scripts/bench_serving.py --backend transformers \
        --model models/pii-1.5b-merged --config-name fp16-mps-b1 \
        --limit 40 --out reports/bench/fp16_mps_b1.json

    # latency config against llama-server
    python scripts/bench_serving.py --backend openai \
        --base-url http://localhost:8080/v1 --config-name q4km-metal-b1 \
        --concurrency 1 --out reports/bench/q4km_b1.json

    # throughput config
    python scripts/bench_serving.py --backend openai \
        --base-url http://localhost:8080/v1 --config-name q4km-metal-b16 \
        --concurrency 16 --out reports/bench/q4km_b16.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord

# ---------------------------------------------------------------------------
# Cost model — mirrors scripts/run_economics.py DEFAULT_PRICING exactly.
# Duplicated deliberately: this harness must produce a self-contained artifact,
# and a silent drift between the two would be invisible in the report. The
# assertion in `_check_cost_model_parity` fails loudly if they diverge.
# ---------------------------------------------------------------------------
COST_MODEL = {
    "student_hardware_usd": 1599.0,
    "student_life_years": 4.0,
    "student_busy_hours_per_day": 8.0,
    "student_watts": 22.0,
    "student_usd_per_kwh": 0.12,
}


def _check_cost_model_parity() -> str | None:
    """Return a warning if run_economics.py's pricing has drifted from ours."""
    try:
        from scripts.run_economics import DEFAULT_PRICING  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from run_economics import DEFAULT_PRICING  # type: ignore
        except Exception:
            return "could not import run_economics.DEFAULT_PRICING to cross-check"
    drift = {k: (v, DEFAULT_PRICING.get(k)) for k, v in COST_MODEL.items()
             if DEFAULT_PRICING.get(k) != v}
    if drift:
        return f"COST MODEL DRIFT vs run_economics.py: {drift}"
    return None


def usd_per_1k(sustained_s_per_record: float, watts: float | None = None) -> dict:
    """On-device cost for 1k records: amortized hardware + metered energy.

    The input is *machine*-seconds per record, not per-request latency. Under
    concurrency these differ by roughly the concurrency factor, and using the
    wrong one is the single easiest way to fake this gate.
    """
    cm = dict(COST_MODEL)
    if watts is not None:
        cm["student_watts"] = watts
    busy_hours = cm["student_life_years"] * 365 * cm["student_busy_hours_per_day"]
    hw_per_hour = cm["student_hardware_usd"] / busy_hours
    machine_hours_per_1k = sustained_s_per_record * 1000 / 3600
    hw = hw_per_hour * machine_hours_per_1k
    energy = cm["student_watts"] / 1000 * machine_hours_per_1k * cm["student_usd_per_kwh"]
    return {
        "usd_per_1k": hw + energy,
        "hardware_usd_per_1k": hw,
        "energy_usd_per_1k": energy,
        "machine_hours_per_1k": machine_hours_per_1k,
        "hardware_usd_per_hour": hw_per_hour,
        "watts": cm["student_watts"],
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One record's outcome. `latency_s` is the request round trip, which is not
    the same as the machine cost of that record when requests overlap."""

    rec_id: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    schema_valid: bool = False
    error: str | None = None
    pred: PIIRecord | None = None
    ttft_s: float | None = None


@dataclass
class Run:
    samples: list[Sample] = field(default_factory=list)
    wall_clock_s: float = 0.0


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = round(q * (len(s) - 1))
    return s[idx]


def load_gold(path: Path, limit: int | None) -> list[PIIRecord]:
    recs = [PIIRecord.model_validate_json(ln)
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return recs[:limit] if limit else recs


# --- backend: OpenAI-compatible endpoint -----------------------------------


def run_openai(args, gold: list[PIIRecord]) -> Run:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0,
                    timeout=args.timeout)

    def one(rec: PIIRecord) -> Sample:
        messages = build_messages(rec.text)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            lat = time.perf_counter() - t0
            usage = getattr(resp, "usage", None)
            raw = resp.choices[0].message.content or ""
            pred, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
            return Sample(
                rec_id=rec.id,
                latency_s=lat,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
                schema_valid=valid,
                pred=pred,
            )
        except Exception as e:  # noqa: BLE001
            return Sample(rec_id=rec.id, latency_s=time.perf_counter() - t0,
                          error=f"{type(e).__name__}: {e}",
                          pred=PIIRecord(id=rec.id, text=rec.text, spans=[], split=rec.split))

    # Warm the server: first request pays Metal kernel compilation and page-in,
    # which is a one-time startup cost and not part of steady-state serving.
    for i in range(args.warmup):
        one(gold[i % len(gold)])

    t_start = time.perf_counter()
    if args.concurrency <= 1:
        samples = [one(r) for r in gold]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            samples = list(ex.map(one, gold))
    wall = time.perf_counter() - t_start
    return Run(samples=samples, wall_clock_s=wall)


# --- backend: local transformers (the incumbent baseline) ------------------


def run_transformers(args, gold: list[PIIRecord]) -> Run:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_mps else torch.bfloat16
    print(f"loading {args.model} (device={device}, dtype={dtype})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    def one(rec: PIIRecord) -> Sample:
        messages = build_messages(rec.text)
        t0 = time.perf_counter()
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt")
        input_ids = enc["input_ids"] if hasattr(enc, "input_ids") else enc
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        if device == "mps":
            torch.mps.synchronize()
        lat = time.perf_counter() - t0
        n_in = int(input_ids.shape[-1])
        n_out = int(out.shape[-1]) - n_in
        raw = tok.decode(out[0][n_in:], skip_special_tokens=True)
        pred, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
        return Sample(rec_id=rec.id, latency_s=lat, prompt_tokens=n_in,
                      completion_tokens=n_out, schema_valid=valid, pred=pred)

    for i in range(args.warmup):
        one(gold[i % len(gold)])

    t_start = time.perf_counter()
    samples = [one(r) for r in gold]
    wall = time.perf_counter() - t_start
    return Run(samples=samples, wall_clock_s=wall)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(args, run: Run, gold: list[PIIRecord]) -> dict:
    ok = [s for s in run.samples if s.error is None]
    lat = [s.latency_s for s in ok]
    n = len(run.samples)
    comp = sum(s.completion_tokens for s in ok)
    prompt = sum(s.prompt_tokens for s in ok)

    # THE number the cost gate consumes: machine-seconds per record. Derived from
    # wall clock over the whole set, so overlapping requests are counted once.
    sustained = run.wall_clock_s / n if n else 0.0
    econ = usd_per_1k(sustained, watts=args.watts)

    out = {
        "config_name": args.config_name,
        "backend": args.backend,
        "model": args.model,
        "quant": args.quant,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "gold": str(args.gold),
        "n_records": n,
        "n_ok": len(ok),
        "errors": n - len(ok),
        "schema_valid": sum(1 for s in ok if s.schema_valid),
        "wall_clock_s": round(run.wall_clock_s, 4),

        # --- G3 inputs ---
        "sustained_s_per_record": round(sustained, 5),
        "records_per_s": round(1 / sustained, 4) if sustained else 0.0,
        "usd_per_1k": round(econ["usd_per_1k"], 6),
        "cost_breakdown": {k: round(v, 6) for k, v in econ.items()},

        # --- G4 inputs (per-request round trip; inflates under concurrency) ---
        "latency": {
            "mean_s": round(statistics.fmean(lat), 4) if lat else 0.0,
            "p50_s": round(pct(lat, 0.50), 4),
            "p90_s": round(pct(lat, 0.90), 4),
            "p95_s": round(pct(lat, 0.95), 4),
            "p99_s": round(pct(lat, 0.99), 4),
            "min_s": round(min(lat), 4) if lat else 0.0,
            "max_s": round(max(lat), 4) if lat else 0.0,
        },

        "tokens": {
            "prompt_total": prompt,
            "completion_total": comp,
            "prompt_per_record": round(prompt / len(ok), 2) if ok else 0,
            "completion_per_record": round(comp / len(ok), 2) if ok else 0,
            "output_tok_s_aggregate": round(comp / run.wall_clock_s, 2) if run.wall_clock_s else 0,
            "total_tok_s_aggregate": round((comp + prompt) / run.wall_clock_s, 2) if run.wall_clock_s else 0,
        },

        # --- run_economics.py compatibility ---
        # `avg_latency_s` is what that script multiplies out to machine-hours, so
        # it MUST carry the sustained figure, not the mean round trip. Under
        # concurrency 1 the two coincide; under concurrency 16 they differ ~16x.
        "avg_latency_s": round(sustained, 5),
        "avg_latency_s_basis": "wall_clock / n_records (machine-seconds per record)",
        "mean_request_latency_s": round(statistics.fmean(lat), 4) if lat else 0.0,
        "p50_latency_s": round(pct(lat, 0.50), 4),
        "p95_latency_s": round(pct(lat, 0.95), 4),
        "total": n,
        "total_tokens_in": prompt,
        "total_tokens_out": comp,
        "adapter": args.adapter,

        "env": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _sysctl("machdep.cpu.brand_string"),
            "ram_gb": round(int(_sysctl("hw.memsize") or 0) / 1024**3, 1) if _sysctl("hw.memsize") else None,
            "python": platform.python_version(),
            "server_cmd": args.server_cmd,
            "llama_cpp_commit": args.llama_cpp_commit,
        },
        "contention_at_start": args.contention_at_start,
        "contention_at_end": _contention(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    warn = _check_cost_model_parity()
    if warn:
        out["cost_model_warning"] = warn
    return out


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _contention() -> dict:
    """Snapshot machine load and swap.

    This is a shared laptop running the owner's normal applications, not a
    quiesced benchmark host. Decode here is memory-bandwidth bound, so competing
    processes move the numbers. Recording the conditions alongside each result is
    the difference between a measurement and an anecdote — a reader can see
    whether a given figure was taken on a busy machine.
    """
    out: dict = {}
    try:
        load = os.getloadavg()
        out["loadavg_1m"], out["loadavg_5m"], out["loadavg_15m"] = [round(x, 2) for x in load]
    except Exception:  # noqa: BLE001
        pass
    swap = _sysctl("vm.swapusage") or ""
    if swap:
        out["swapusage"] = swap
    out["ncpu"] = _sysctl("hw.ncpu")
    return out


def print_summary(d: dict) -> None:
    L = d["latency"]
    T = d["tokens"]
    print()
    print("=" * 68)
    print(f"  {d['config_name']}   ({d['backend']}, concurrency={d['concurrency']})")
    print("=" * 68)
    print(f"  records                 {d['n_ok']}/{d['n_records']}  "
          f"(errors {d['errors']}, schema-valid {d['schema_valid']})")
    print(f"  wall clock              {d['wall_clock_s']:.2f} s")
    print()
    print("  -- G3 basis: machine cost (overlap counted once) --")
    print(f"  sustained s/record      {d['sustained_s_per_record']:.4f}")
    print(f"  records/s               {d['records_per_s']:.3f}")
    print(f"  output tok/s aggregate  {T['output_tok_s_aggregate']:.1f}")
    print(f"  $ / 1k records          ${d['usd_per_1k']:.5f}")
    print()
    print("  -- G4 basis: per-request round trip --")
    print(f"  mean / p50 / p95 / p99  {L['mean_s']:.3f} / {L['p50_s']:.3f} / "
          f"{L['p95_s']:.3f} / {L['p99_s']:.3f} s")
    print()
    print(f"  tokens/record           {T['prompt_per_record']:.0f} in, "
          f"{T['completion_per_record']:.1f} out")
    if "cost_model_warning" in d:
        print(f"\n  [WARN] {d['cost_model_warning']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure serving latency and sustained throughput (gates G3/G4).")
    ap.add_argument("--backend", choices=["openai", "transformers"], default="openai")
    ap.add_argument("--model", required=True, help="Model id (server) or local path (transformers)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (transformers backend only)")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="Use only first N records")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=2, help="Warmup requests, excluded from timing")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the run N times; report the best-throughput pass")
    ap.add_argument("--config-name", required=True, help="Name stamped into the artifact")
    ap.add_argument("--quant", default=None, help="Quantization label for the record")
    ap.add_argument("--watts", type=float, default=None,
                    help="Override sustained package watts for the energy term")
    ap.add_argument("--server-cmd", default=None, help="Server command line, recorded for provenance")
    ap.add_argument("--llama-cpp-commit", default=None, help="llama.cpp commit, recorded for provenance")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON artifact here")
    ap.add_argument("--save-predictions", type=Path, default=None,
                    help="Also write predictions JSONL (for scoring with run_eval.py)")
    args = ap.parse_args()

    args.contention_at_start = _contention()

    gold = load_gold(args.gold, args.limit)
    if not gold:
        print("no gold records loaded", file=sys.stderr)
        return 1
    print(f"{args.config_name}: {len(gold)} records, backend={args.backend}, "
          f"concurrency={args.concurrency}", file=sys.stderr)

    runner = run_openai if args.backend == "openai" else run_transformers
    best: dict | None = None
    best_run: Run | None = None
    for i in range(args.repeat):
        run = runner(args, gold)
        d = summarize(args, run, gold)
        if best is None or d["records_per_s"] > best["records_per_s"]:
            best, best_run = d, run
        if args.repeat > 1:
            print(f"  pass {i+1}/{args.repeat}: {d['records_per_s']:.3f} rec/s", file=sys.stderr)
    assert best is not None and best_run is not None
    best["repeat_passes"] = args.repeat

    print_summary(best)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        by_id = {s.rec_id: s for s in best_run.samples}
        with args.save_predictions.open("w", encoding="utf-8") as fh:
            for rec in gold:
                s = by_id.get(rec.id)
                pred = s.pred if s and s.pred else PIIRecord(
                    id=rec.id, text=rec.text, spans=[], split=rec.split)
                fh.write(pred.model_dump_json() + "\n")
        meta = args.save_predictions.with_suffix(".meta.json")
        meta.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.save_predictions} (+ .meta.json)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
