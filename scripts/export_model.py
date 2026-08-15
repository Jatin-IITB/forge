#!/usr/bin/env python3
"""Merge the LoRA adapter into the base model and export deployable artifacts.

The adapter is not the product — a merged, quantized, offline-runnable model is.
This script produces that, in the order a deployment actually needs:

    merge   adapter + base -> a standalone HF model directory
    gguf    HF model -> GGUF (llama.cpp), the on-device/air-gapped artifact
    awq     HF model -> AWQ 4-bit (requires CUDA; see --help for why)

Quantization is NOT trusted until it re-passes the gates. A quantized model that
drops below the parity floor does not ship, so `make forge` re-runs the eval
harness against each artifact rather than assuming quantization is lossless.

Usage:
    python scripts/export_model.py merge \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter checkpoints/run_002/final \
        --output models/pii-1.5b-merged

    python scripts/export_model.py gguf \
        --merged models/pii-1.5b-merged \
        --output models/pii-1.5b-gguf \
        --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_merge(args: argparse.Namespace) -> int:
    """Fold LoRA weights into the base model so the result stands alone."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"writing merged model -> {args.output}")
    model.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))

    provenance = {
        "base_model": args.base,
        "adapter": str(args.adapter),
        "merged_dtype": "float16",
    }
    adapter_meta = args.adapter.parent / "meta.json"
    if adapter_meta.exists():
        provenance["training"] = json.loads(adapter_meta.read_text(encoding="utf-8"))
    (args.output / "forge_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print("merged. NOTE: this artifact is unverified until it re-passes the gates:")
    print(f"  python scripts/run_inference.py data/gold/test.jsonl preds.jsonl --model {args.output}")
    return 0


def _find_llama_cpp(explicit: Path | None) -> Path | None:
    """Locate a llama.cpp checkout containing the HF->GGUF converter."""
    candidates = [explicit] if explicit else []
    candidates += [
        Path("llama.cpp"),
        Path.home() / "llama.cpp",
        Path.home() / "src" / "llama.cpp",
    ]
    for c in candidates:
        if c and (c / "convert_hf_to_gguf.py").exists():
            return c
    return None


def cmd_gguf(args: argparse.Namespace) -> int:
    """Convert to GGUF and quantize — the artifact that runs offline on a laptop."""
    repo = _find_llama_cpp(args.llama_cpp)
    if repo is None:
        print(
            "llama.cpp not found. Clone it once, then re-run:\n"
            "    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "    cmake -B ~/llama.cpp/build -S ~/llama.cpp && "
            "cmake --build ~/llama.cpp/build --config Release -j\n"
            "Or pass --llama-cpp /path/to/llama.cpp",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    f16_path = args.output / "model-f16.gguf"

    print(f"converting to GGUF f16 via {repo}")
    conv = subprocess.run(
        [
            sys.executable,
            str(repo / "convert_hf_to_gguf.py"),
            str(args.merged),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ],
        check=False,
    )
    if conv.returncode != 0:
        print("GGUF conversion failed", file=sys.stderr)
        return conv.returncode

    quantize_bin = None
    for cand in (repo / "build" / "bin" / "llama-quantize", repo / "llama-quantize"):
        if cand.exists():
            quantize_bin = cand
            break
    if quantize_bin is None:
        print(
            f"f16 GGUF written to {f16_path}, but llama-quantize binary not found — "
            "build llama.cpp to produce the quantized artifact.",
            file=sys.stderr,
        )
        return 1

    for quant in args.quant:
        out = args.output / f"model-{quant}.gguf"
        print(f"quantizing -> {quant}")
        q = subprocess.run([str(quantize_bin), str(f16_path), str(out), quant], check=False)
        if q.returncode != 0:
            print(f"quantization to {quant} failed", file=sys.stderr)
            return q.returncode
        size_mb = out.stat().st_size / 1e6
        print(f"  {out.name}: {size_mb:.0f} MB")

    if not args.keep_f16:
        f16_path.unlink()

    print("\nGGUF artifacts are UNVERIFIED until they re-pass the gates.")
    print("Run them through Ollama or llama-server, then score with run_eval.py.")
    return 0


def cmd_awq(args: argparse.Namespace) -> int:
    """AWQ 4-bit quantization. Requires CUDA — documented, not silently skipped."""
    try:
        import torch
    except ImportError:
        print("torch not available", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print(
            "AWQ requires CUDA and this machine has none.\n"
            "\n"
            "This is a hardware limit, not a code gap. AWQ's calibration pass runs\n"
            "fused CUDA kernels; there is no Apple Silicon path. Options:\n"
            "  1. Rent a GPU for ~15 minutes (any T4/A10 instance, roughly $1) and\n"
            "     run this same command there.\n"
            "  2. Ship GGUF only — it is the better artifact for the on-device story\n"
            "     anyway, and covers the same deployment claim.\n"
            "\n"
            "Whichever you choose, record it: an unrun quantization must not be\n"
            "described as a shipped one.",
            file=sys.stderr,
        )
        return 2

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"loading {args.merged} for AWQ quantization")
    model = AutoAWQForCausalLM.from_pretrained(str(args.merged))
    tokenizer = AutoTokenizer.from_pretrained(str(args.merged), trust_remote_code=True)

    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"wrote AWQ model -> {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export deployable model artifacts.")
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    m.add_argument("--base", required=True, help="Base model name/path")
    m.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir")
    m.add_argument("--output", type=Path, required=True, help="Merged model output dir")
    m.set_defaults(func=cmd_merge)

    g = sub.add_parser("gguf", help="Convert merged model to quantized GGUF")
    g.add_argument("--merged", type=Path, required=True, help="Merged HF model dir")
    g.add_argument("--output", type=Path, required=True, help="GGUF output dir")
    g.add_argument(
        "--quant", nargs="+", default=["Q4_K_M", "Q8_0"],
        help="Quantization types (default: Q4_K_M Q8_0)",
    )
    g.add_argument("--llama-cpp", type=Path, default=None, help="Path to llama.cpp checkout")
    g.add_argument("--keep-f16", action="store_true", help="Keep the intermediate f16 GGUF")
    g.set_defaults(func=cmd_gguf)

    a = sub.add_parser("awq", help="AWQ 4-bit quantization (requires CUDA)")
    a.add_argument("--merged", type=Path, required=True, help="Merged HF model dir")
    a.add_argument("--output", type=Path, required=True, help="AWQ output dir")
    a.set_defaults(func=cmd_awq)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
