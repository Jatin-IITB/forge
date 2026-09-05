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
    token-classifier
                  local one-pass BIOES head on MPS/CUDA/CPU, batched

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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from forge.grammar import (
    compact_spans_gbnf,
    compact_spans_json_schema,
    line_spans_gbnf,
)
from forge.inference import build_messages, parse_line_response, parse_response
from forge.schema import PIIRecord

# ---------------------------------------------------------------------------
# Cost model — mirrors scripts/run_economics.py DEFAULT_PRICING exactly.
# Duplicated deliberately: this harness must produce a self-contained artifact,
# and a silent drift between the two would be invisible in the report. The
# assertion in `_check_cost_model_parity` fails loudly if they diverge.
# ---------------------------------------------------------------------------
# Defaults describe the CURRENT reference machine: an ASUS Vivobook Pro 15 with
# an RTX 3050 Ti Laptop GPU, bought for INR 85,000. Converted at 83 INR/USD --
# the rate giving the HIGHEST dollar figure of the plausible range, because a
# more expensive machine makes our own cost ratio worse. That is the same
# direction as pricing the teacher at paid rates when we measured on the free
# tier: every judgement call is taken against ourselves.
#
# Artifacts written before 2026-09-05 used a 16 GB M1 ($1599, 22 W) and are not
# comparable on cost. Every artifact now records `machine` and the effective
# model, so a figure cannot be silently read against the wrong hardware.
COST_MODEL = {
    "student_hardware_usd": 1024.0,
    "student_life_years": 4.0,
    "student_busy_hours_per_day": 8.0,
    # Sustained wall draw under inference load. 120 W is the adapter rating,
    # used as an upper bound until a measured figure replaces it -- again the
    # conservative direction, since overstating draw overstates our cost.
    "student_watts": 120.0,
    "student_usd_per_kwh": 0.12,
}
DEFAULT_MACHINE = "asus-vivobook-pro-15-rtx3050ti"


def _check_cost_model_parity() -> str | None:
    """Return a warning if run_economics.py's pricing has drifted from ours."""
    try:
        from scripts.run_economics import DEFAULT_PRICING  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from run_economics import DEFAULT_PRICING  # type: ignore
        except Exception:  # noqa: BLE001
            return "could not import run_economics.DEFAULT_PRICING to cross-check"
    drift = {k: (v, DEFAULT_PRICING.get(k)) for k, v in COST_MODEL.items()
             if DEFAULT_PRICING.get(k) != v}
    if drift:
        return f"COST MODEL DRIFT vs run_economics.py: {drift}"
    return None


def usd_per_1k(
    sustained_s_per_record: float,
    watts: float | None = None,
    hardware_usd: float | None = None,
    machine: str | None = None,
) -> dict:
    """On-device cost for 1k records: amortized hardware + metered energy.

    The input is *machine*-seconds per record, not per-request latency. Under
    concurrency these differ by roughly the concurrency factor, and using the
    wrong one is the single easiest way to fake this gate.

    `hardware_usd` and `machine` exist because the second-easiest way to fake it
    is quoting one machine's throughput against another machine's price. The
    reference machine changed once already, and the two differ enough to move G3
    by ~2x on the hourly rate alone, so the effective model and the machine label
    travel with every number.
    """
    cm = dict(COST_MODEL)
    if watts is not None:
        cm["student_watts"] = watts
    if hardware_usd is not None:
        cm["student_hardware_usd"] = hardware_usd
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
        # Carried so a cost figure can never be read against the wrong laptop.
        "machine": machine or DEFAULT_MACHINE,
        "hardware_usd": cm["student_hardware_usd"],
        "life_years": cm["student_life_years"],
        "busy_hours_per_day": cm["student_busy_hours_per_day"],
        "usd_per_kwh": cm["student_usd_per_kwh"],
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
    attempts: list[dict] = field(default_factory=list)


@dataclass
class Run:
    samples: list[Sample] = field(default_factory=list)
    wall_clock_s: float = 0.0
    padded_input_tokens: int = 0
    source_tokens: int = 0
    runtime: dict = field(default_factory=dict)


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

    def extra_body_for(constraint: str) -> dict | None:
        if constraint == "gbnf":
            return {"grammar": compact_spans_gbnf()}
        if constraint == "json-schema":
            return {"json_schema": compact_spans_json_schema()}
        if constraint == "line-gbnf":
            return {"grammar": line_spans_gbnf()}
        return None

    first_constraint = (
        "line-gbnf"
        if args.line_grammar
        else "gbnf"
        if args.compact_grammar
        else args.compact_constraint
    )

    def one(rec: PIIRecord) -> Sample:
        messages = build_messages(
            rec.text,
            compact=args.compact_prompt,
            line=args.line_prompt,
        )
        t0 = time.perf_counter()
        try:
            attempts: list[dict] = []

            def request(constraint: str):
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    extra_body=extra_body_for(constraint),
                )
                usage = getattr(resp, "usage", None)
                raw = resp.choices[0].message.content or ""
                parser = parse_line_response if args.line_prompt else parse_response
                pred, valid = parser(rec.id, rec.text, raw, split=rec.split)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
                attempts.append({
                    "constraint": constraint,
                    "schema_valid": valid,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw": raw,
                })
                return pred, valid, prompt_tokens, completion_tokens

            pred, valid, prompt_tokens, completion_tokens = request(first_constraint)
            if not valid and args.retry_invalid != "none":
                pred, valid, retry_prompt, retry_completion = request(args.retry_invalid)
                prompt_tokens += retry_prompt
                completion_tokens += retry_completion

            lat = time.perf_counter() - t0
            return Sample(
                rec_id=rec.id,
                latency_s=lat,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                schema_valid=valid,
                pred=pred,
                attempts=attempts,
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
        messages = build_messages(
            rec.text,
            compact=args.compact_prompt,
            line=args.line_prompt,
        )
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
        parser = parse_line_response if args.line_prompt else parse_response
        pred, valid = parser(rec.id, rec.text, raw, split=rec.split)
        return Sample(rec_id=rec.id, latency_s=lat, prompt_tokens=n_in,
                      completion_tokens=n_out, schema_valid=valid, pred=pred)

    for i in range(args.warmup):
        one(gold[i % len(gold)])

    t_start = time.perf_counter()
    samples = [one(r) for r in gold]
    wall = time.perf_counter() - t_start
    return Run(samples=samples, wall_clock_s=wall)


# --- backend: local one-pass BIOES token classifier ------------------------


def run_token_classifier(args, gold: list[PIIRecord]) -> Run:
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from forge.token_classifier import constrained_viterbi_batch, decode_bioes
    from forge.token_model import ForgeQwen2ForTokenClassification

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    dtype = torch.float16 if device == "mps" else torch.bfloat16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(args.model)
    config._attn_implementation = "eager"
    print(f"loading {args.model} (device={device}, dtype={dtype})", file=sys.stderr)
    model = ForgeQwen2ForTokenClassification.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
    ).to(device)
    model.eval()

    def rendered_input(record: PIIRecord) -> tuple[str, int]:
        if args.token_input == "raw":
            return record.text, 0
        rendered = tok.apply_chat_template(
            build_messages(record.text),
            tokenize=False,
            add_generation_prompt=False,
        )
        source_start = rendered.rfind(record.text)
        if source_start < 0:
            raise ValueError(f"{record.id}: source text not found in rendered system prompt")
        return rendered, source_start

    def run_once(records: list[PIIRecord]) -> Run:
        samples: list[Sample] = []
        padded_input_tokens = 0
        source_tokens = 0
        started = time.perf_counter()
        ordered_records = (
            sorted(records, key=lambda record: len(record.text))
            if args.length_bucket
            else records
        )
        for batch_start in range(0, len(ordered_records), args.batch_size):
            batch_started = time.perf_counter()
            batch = ordered_records[batch_start : batch_start + args.batch_size]
            rendered = [rendered_input(record) for record in batch]
            encoded = tok(
                [item[0] for item in rendered],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            attention_mask = encoded["attention_mask"]
            padded_input_tokens += int(encoded["input_ids"].numel())
            model_inputs = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = model(**model_inputs).logits
            if device == "mps":
                torch.mps.synchronize()

            logits_cpu = logits.float().cpu().numpy()
            source_logits: list[np.ndarray] = []
            local_offsets_by_record: list[list[tuple[int, int]]] = []
            input_lengths: list[int] = []
            for index, record in enumerate(batch):
                input_length = int(attention_mask[index].sum())
                input_lengths.append(input_length)
                source_start = rendered[index][1]
                source_end = source_start + len(record.text)
                token_indices = [
                    token_index
                    for token_index, (start, end) in enumerate(
                        offsets[index, :input_length].tolist()
                    )
                    if end > start and end > source_start and start < source_end
                ]
                if not token_indices:
                    raise ValueError(f"{record.id}: no source tokens found in classifier input")
                source_tokens += len(token_indices)
                source_logits.append(logits_cpu[index, token_indices])
                local_offsets = [
                    (
                        max(0, int(offsets[index, token_index, 0]) - source_start),
                        min(
                            len(record.text),
                            int(offsets[index, token_index, 1]) - source_start,
                        ),
                    )
                    for token_index in token_indices
                ]
                if max(end for _, end in local_offsets) < len(record.text):
                    raise ValueError(
                        f"{record.id}: max length truncated the classifier source text"
                    )
                local_offsets_by_record.append(local_offsets)

            source_lengths = [len(row) for row in source_logits]
            padded_source_logits = np.zeros(
                (len(batch), max(source_lengths), logits_cpu.shape[2]),
                dtype=np.float32,
            )
            for index, row in enumerate(source_logits):
                padded_source_logits[index, : len(row)] = row
            paths = constrained_viterbi_batch(padded_source_logits, source_lengths)

            decoded_batch: list[tuple[PIIRecord, int]] = []
            for record, local_offsets, path, input_length in zip(
                batch,
                local_offsets_by_record,
                paths,
                input_lengths,
            ):
                decoded_batch.append(
                    (
                        decode_bioes(
                            record.id,
                            record.text,
                            local_offsets,
                            path,
                            split=record.split,
                        ),
                        input_length,
                    )
                )
            batch_elapsed = time.perf_counter() - batch_started
            samples.extend(
                Sample(
                    rec_id=record.id,
                    latency_s=batch_elapsed,
                    prompt_tokens=input_length,
                    schema_valid=True,
                    pred=prediction,
                )
                for record, (prediction, input_length) in zip(batch, decoded_batch)
            )
        return Run(
            samples=samples,
            wall_clock_s=time.perf_counter() - started,
            padded_input_tokens=padded_input_tokens,
            source_tokens=source_tokens,
            runtime={
                "device": device,
                "dtype": str(dtype),
                "batch_size": args.batch_size,
                "token_input": args.token_input,
                "length_bucket": args.length_bucket,
                "max_length": args.max_length,
            },
        )

    for _ in range(args.warmup):
        run_once(gold[: args.batch_size])
    return run_once(gold)


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
    econ = usd_per_1k(sustained, watts=args.watts,
                      hardware_usd=args.hardware_usd, machine=args.machine)

    out = {
        "config_name": args.config_name,
        "backend": args.backend,
        "model": args.model,
        "quant": args.quant,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size if args.backend == "token-classifier" else None,
        "token_input": args.token_input if args.backend == "token-classifier" else None,
        "length_bucket": args.length_bucket if args.backend == "token-classifier" else False,
        "max_length": args.max_length if args.backend == "token-classifier" else None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "compact_prompt": args.compact_prompt,
        "line_prompt": args.line_prompt,
        "line_grammar": args.line_grammar,
        "compact_grammar": args.compact_grammar,
        "compact_constraint": args.compact_constraint,
        "retry_invalid": args.retry_invalid,
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
        # `machine` is a string, and rounding the dict blindly crashed on it the
        # first time this ran after that field was added. Round the numbers, pass
        # the label through -- dropping it would defeat the reason it is carried.
        "cost_breakdown": {
            k: round(v, 6) if isinstance(v, (int, float)) else v
            for k, v in econ.items()
        },

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
            "source_total": run.source_tokens,
            "source_per_record": round(run.source_tokens / len(ok), 2) if ok else 0,
            "padded_input_total": run.padded_input_tokens,
            "padded_input_per_record": (
                round(run.padded_input_tokens / len(ok), 2) if ok else 0
            ),
            "padded_input_tok_s_aggregate": (
                round(run.padded_input_tokens / run.wall_clock_s, 2)
                if run.wall_clock_s
                else 0
            ),
            "output_tok_s_aggregate": round(comp / run.wall_clock_s, 2) if run.wall_clock_s else 0,
            "total_tok_s_aggregate": round((comp + prompt) / run.wall_clock_s, 2) if run.wall_clock_s else 0,
        },
        "retry": {
            "records_retried": sum(1 for s in ok if len(s.attempts) > 1),
            "request_attempts_total": sum(len(s.attempts) for s in ok),
            "details": [
                {"rec_id": s.rec_id, "attempts": s.attempts}
                for s in ok if len(s.attempts) > 1
            ],
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
        "runtime": run.runtime,
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
    except Exception:  # noqa: BLE001
        return None


def _contention() -> dict:
    """Snapshot machine load and swap.

    This is a shared laptop running the owner's normal applications, not a
    quiesced benchmark host. Decode here is memory-bandwidth bound, so competing
    processes move the numbers. Recording the conditions alongside each result is
    the difference between a measurement and an anecdote — a reader can see
    whether a given figure was taken on a busy machine.

    Cross-platform since 2026-09-05. ``os.getloadavg`` is POSIX-only and raises
    AttributeError on Windows rather than OSError, so the original guard missed
    it and every benchmark died before its first record. ``sysctl`` is likewise
    macOS-only. Contention is context, never a result, so a platform that cannot
    supply a field omits it instead of failing the run.
    """
    out: dict = {}

    load = None
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        try:
            import psutil  # transitive dep of accelerate; simulated on Windows

            load = psutil.getloadavg()
        except Exception:  # noqa: BLE001
            load = None
    if load:
        out["loadavg_1m"], out["loadavg_5m"], out["loadavg_15m"] = [round(x, 2) for x in load]

    swap = _sysctl("vm.swapusage")
    if swap:
        out["swapusage"] = swap

    ncpu = _sysctl("hw.ncpu") or str(os.cpu_count() or "")
    if ncpu:
        out["ncpu"] = ncpu

    # Where sysctl is absent, report memory pressure through psutil instead.
    # The question this field answers is "was the machine under load", and a
    # swap percentage answers it as well as macOS's swapusage string does.
    if "swapusage" not in out:
        try:
            import psutil

            out["mem_percent"] = psutil.virtual_memory().percent
            out["swap_percent"] = psutil.swap_memory().percent
        except Exception:  # noqa: BLE001, S110 -- context is optional, never a result
            pass

    out["platform"] = sys.platform
    return out


def print_summary(d: dict) -> None:
    L = d["latency"]
    T = d["tokens"]
    parallelism = (
        f"batch={d['batch_size']}"
        if d["backend"] == "token-classifier"
        else f"concurrency={d['concurrency']}"
    )
    print()
    print("=" * 68)
    print(f"  {d['config_name']}   ({d['backend']}, {parallelism})")
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
    if T["padded_input_total"]:
        print(
            f"  padded/source tokens    {T['padded_input_per_record']:.1f} / "
            f"{T['source_per_record']:.1f} per record"
        )
    if "cost_model_warning" in d:
        print(f"\n  [WARN] {d['cost_model_warning']}")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Split out from main() so tests can construct real args.

    `summarize` reads ~30 attributes off the Namespace. A test that hand-builds
    one duplicates the CLI and drifts from it, which is the same defect class as
    the crash this file's tests exist to prevent -- so the test takes its
    defaults from here instead.
    """
    ap = argparse.ArgumentParser(
        description="Measure serving latency and sustained throughput (gates G3/G4).")
    ap.add_argument(
        "--backend",
        choices=["openai", "transformers", "token-classifier"],
        default="openai",
    )
    ap.add_argument("--model", required=True, help="Model id (server) or local path (transformers)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (transformers backend only)")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="Use only first N records")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the token-classifier backend",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device for the token-classifier backend",
    )
    ap.add_argument(
        "--token-input",
        choices=["raw", "system"],
        default="raw",
        help="Feed raw source text or the legacy instruction prompt to the token classifier",
    )
    ap.add_argument(
        "--length-bucket",
        action="store_true",
        help="Batch token-classifier records by source character length to reduce padding",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer truncation length for the token-classifier backend",
    )
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Request the compact s/l/t response shape",
    )
    ap.add_argument(
        "--line-prompt",
        action="store_true",
        help="Request LABEL<TAB><JSON-string> lines from a line-target checkpoint",
    )
    ap.add_argument(
        "--line-grammar",
        action="store_true",
        help="Force the escaped line protocol (OpenAI backend only)",
    )
    ap.add_argument(
        "--compact-grammar",
        action="store_true",
        help="Force the compact s/l/t response shape (OpenAI backend only)",
    )
    ap.add_argument(
        "--compact-constraint",
        choices=["none", "gbnf", "json-schema"],
        default="none",
        help="Constraint for the first compact request; json-schema uses LLGuidance when built in",
    )
    ap.add_argument(
        "--retry-invalid",
        choices=["none", "gbnf", "json-schema", "line-gbnf"],
        default="none",
        help="Retry an invalid alternate-format response once with this constraint",
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=2, help="Warmup requests, excluded from timing")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the run N times; selection is controlled separately",
    )
    ap.add_argument(
        "--repeat-selection",
        choices=["first", "median", "best"],
        default="best",
        help="Which measured pass is the headline result; use median to avoid best-of-N",
    )
    ap.add_argument("--config-name", required=True, help="Name stamped into the artifact")
    ap.add_argument("--quant", default=None, help="Quantization label for the record")
    ap.add_argument("--watts", type=float, default=None,
                    help="Override sustained package watts for the energy term")
    ap.add_argument("--hardware-usd", type=float, default=None,
                    help="Purchase price of the machine being measured. Defaults to "
                         "the current reference laptop; set it when benchmarking "
                         "different hardware so the cost is not attributed wrongly")
    ap.add_argument("--machine", default=None,
                    help="Machine label recorded in the artifact, e.g. "
                         "'asus-vivobook-pro-15-rtx3050ti'")
    ap.add_argument("--server-cmd", default=None, help="Server command line, recorded for provenance")
    ap.add_argument("--llama-cpp-commit", default=None, help="llama.cpp commit, recorded for provenance")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON artifact here")
    ap.add_argument("--save-predictions", type=Path, default=None,
                    help="Also write predictions JSONL (for scoring with run_eval.py)")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if (
        args.compact_grammar
        or args.line_grammar
        or args.compact_constraint != "none"
        or args.retry_invalid != "none"
    ) and args.backend != "openai":
        ap.error("compact constraints and retries require the openai backend")
    if args.compact_grammar and args.compact_constraint != "none":
        ap.error("--compact-grammar is an alias for --compact-constraint gbnf; use only one")
    if args.retry_invalid in {"gbnf", "json-schema"} and not args.compact_prompt:
        ap.error("compact --retry-invalid modes require --compact-prompt")
    if args.retry_invalid == "line-gbnf" and not args.line_prompt:
        ap.error("--retry-invalid line-gbnf requires --line-prompt")
    if args.line_grammar and not args.line_prompt:
        ap.error("--line-grammar requires --line-prompt")
    if args.compact_prompt and args.line_prompt:
        ap.error("--compact-prompt and --line-prompt are mutually exclusive")
    if args.backend != "token-classifier" and (
        args.device != "auto"
        or args.token_input != "raw"
        or args.length_bucket
        or args.max_length != 512
    ):
        ap.error(
            "--device, --token-input, --length-bucket, and --max-length are token-classifier options"
        )
    if args.backend == "token-classifier" and args.concurrency != 1:
        ap.error("token-classifier uses --batch-size; leave --concurrency at 1")
    if args.repeat_selection == "median" and args.repeat % 2 == 0:
        ap.error("--repeat-selection median requires an odd --repeat")

    gold = load_gold(args.gold, args.limit)
    if not gold:
        print("no gold records loaded", file=sys.stderr)
        return 1
    print(f"{args.config_name}: {len(gold)} records, backend={args.backend}, "
          f"concurrency={args.concurrency}", file=sys.stderr)

    runner = {
        "openai": run_openai,
        "transformers": run_transformers,
        "token-classifier": run_token_classifier,
    }[args.backend]
    measured: list[tuple[dict, Run]] = []
    for i in range(args.repeat):
        args.contention_at_start = _contention()
        run = runner(args, gold)
        d = summarize(args, run, gold)
        measured.append((d, run))
        if args.repeat > 1:
            print(f"  pass {i+1}/{args.repeat}: {d['records_per_s']:.3f} rec/s", file=sys.stderr)
    if args.repeat_selection == "first":
        selected, selected_run = measured[0]
    elif args.repeat_selection == "median":
        selected, selected_run = sorted(
            measured, key=lambda item: item[0]["records_per_s"]
        )[len(measured) // 2]
    else:
        selected, selected_run = max(
            measured, key=lambda item: item[0]["records_per_s"]
        )
    selected["repeat_passes"] = args.repeat
    selected["repeat_selection"] = args.repeat_selection
    selected["repeat_summaries"] = [
        {
            "pass": index,
            "wall_clock_s": result["wall_clock_s"],
            "sustained_s_per_record": result["sustained_s_per_record"],
            "records_per_s": result["records_per_s"],
            "contention_at_start": result["contention_at_start"],
            "contention_at_end": result["contention_at_end"],
        }
        for index, (result, _) in enumerate(measured, 1)
    ]

    print_summary(selected)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        by_id = {s.rec_id: s for s in selected_run.samples}
        with args.save_predictions.open("w", encoding="utf-8") as fh:
            for rec in gold:
                s = by_id.get(rec.id)
                pred = s.pred if s and s.pred else PIIRecord(
                    id=rec.id, text=rec.text, spans=[], split=rec.split)
                fh.write(pred.model_dump_json() + "\n")
        meta = args.save_predictions.with_suffix(".meta.json")
        meta.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.save_predictions} (+ .meta.json)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
