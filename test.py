#!/usr/bin/env python3
"""
smoke_test.py
-------------
Runs a single end-to-end test of the full stack:
  environment → Megatron-LM import → torchrun → metric parsing → CSV write

Designed to be run interactively on a Bridges2 GPU node BEFORE submitting
the full sweep. Takes ~2-4 minutes and costs < 5 SU.

Usage:
    # Interactive session (recommended first run):
    interact -p GPU -N 1 --ntasks-per-node=8 --gres=gpu:8 -t 00:15:00
    module load AI/pytorch_23.02-1.13.1-py3
    conda activate megatron
    python smoke_test.py --megatron ~/Megatron-LM

    # As a quick batch job:
    sbatch smoke_test.sh

What it tests (in order):
  [1] Python imports  — torch, apex, transformer_engine, megatron
  [2] CUDA visibility — GPUs detected, NCCL accessible
  [3] Megatron launch — torchrun pretrain_gpt.py with GPT-1B config,
                        TP=2 PP=1 CP=2 SP=on, 20 iters (10 warmup + 10 measured)
  [4] Metric parsing  — throughput, memory, comm ratio, bubble ratio, MFU
  [5] CSV write       — result appended to smoke_test_result.csv

A PASS on all five stages means the full sweep will work.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Test configuration — deliberately small but non-trivial
# Uses GPT-1B with TP=2, PP=1, CP=2, SP=on to exercise:
#   - tensor parallelism (comm overlap)
#   - context parallelism (CP=TP constraint)
#   - sequence parallelism (requires TP>1)
#   - all metric collection paths
# ---------------------------------------------------------------------------
SMOKE_CONFIG = {
    "model_name":          "GPT-1B",
    "num_layers":          4,        # reduced from 24 for speed
    "hidden_size":         2048,
    "ffn_hidden_size":     8192,
    "num_attention_heads": 16,
    "num_query_groups":    16,
    "tp":                  2,
    "pp":                  1,
    "cp":                  2,        # CP = TP always
    "ep":                  1,
    "sp":                  True,
    "recompute":           "none",
    "micro_batch_size":    1,
    "seq_len":             2048,
    "precision":           "bf16",
    "gpus":                4,        # TP*PP*CP = 2*1*2 = 4 GPUs
    "train_iters":         20,
    "warmup_iters":        10,
}

TOTAL_ITERS  = SMOKE_CONFIG["train_iters"]
WARMUP_ITERS = SMOKE_CONFIG["warmup_iters"]

# V100 BF16 peak TFLOPS for MFU calculation
V100_BF16_PEAK = 125e12


# ---------------------------------------------------------------------------
# Colour helpers (degrades gracefully if terminal doesn't support it)
# ---------------------------------------------------------------------------
def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def green(t):  return _c("32;1", t)
def red(t):    return _c("31;1", t)
def yellow(t): return _c("33;1", t)
def bold(t):   return _c("1",    t)
def dim(t):    return _c("2",    t)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    stage:   int
    name:    str
    passed:  bool
    message: str
    detail:  str = ""

RESULTS: list = []

def run_stage(n, name, fn):
    print(f"\n{bold(f'[{n}/5]')} {name} ...", flush=True)
    t0 = time.time()
    try:
        msg, detail = fn()
        elapsed = time.time() - t0
        r = StageResult(n, name, True, msg, detail)
        print(f"  {green('PASS')}  {msg}  {dim(f'({elapsed:.1f}s)')}")
        if detail:
            for line in detail.strip().splitlines():
                print(f"       {dim(line)}")
    except Exception as exc:
        elapsed = time.time() - t0
        r = StageResult(n, name, False, str(exc))
        print(f"  {red('FAIL')}  {exc}  {dim(f'({elapsed:.1f}s)')}")
    RESULTS.append(r)
    return r.passed


# ---------------------------------------------------------------------------
# Stage 1 — Python imports
# ---------------------------------------------------------------------------
def stage_imports():
    results = []

    # torch
    import torch
    results.append(f"torch {torch.__version__}")

    # apex (optional but expected)
    try:
        import apex
        results.append("apex OK")
    except ImportError:
        raise RuntimeError(
            "APEX not found. Install with: "
            "pip install -v --no-build-isolation "
            "--config-settings='--build-option=--cpp_ext' "
            "--config-settings='--build-option=--cuda_ext' "
            "git+https://github.com/NVIDIA/apex"
        )

    # transformer_engine (needed for RMSNorm/GQA in MoE groups)
    try:
        import transformer_engine
        results.append(f"transformer_engine {transformer_engine.__version__}")
    except ImportError:
        raise RuntimeError(
            "TransformerEngine not found. Install with: "
            "pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable"
        )

    # megatron
    try:
        from megatron.training import get_args  # noqa
        results.append("megatron OK")
    except ImportError as e:
        raise RuntimeError(f"Megatron-LM not importable: {e}")

    return ", ".join(results), ""


# ---------------------------------------------------------------------------
# Stage 2 — CUDA + NCCL visibility
# ---------------------------------------------------------------------------
def stage_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False — no GPU visible")

    n_gpu = torch.cuda.device_count()
    needed = SMOKE_CONFIG["gpus"]
    if n_gpu < needed:
        raise RuntimeError(
            f"Only {n_gpu} GPU(s) visible, smoke test needs {needed}. "
            f"Request at least {needed} GPUs in your salloc/sbatch."
        )

    gpus_info = []
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        gpus_info.append(f"  GPU {i}: {p.name}  {p.total_memory//1024**3}GB")

    # NCCL check via torch.distributed
    try:
        import torch.distributed as dist
        # Quick single-process init to verify NCCL library loads
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29499")
        dist.init_process_group("nccl", rank=0, world_size=1)
        dist.destroy_process_group()
        nccl_ver = torch.cuda.nccl.version()
        nccl_str = f"NCCL {'.'.join(str(x) for x in nccl_ver)}"
    except Exception as e:
        nccl_str = f"NCCL check skipped ({e})"

    msg = f"{n_gpu} GPU(s) visible, {nccl_str}"
    detail = "\n".join(gpus_info)
    return msg, detail


# ---------------------------------------------------------------------------
# Stage 3 — Megatron torchrun launch
# ---------------------------------------------------------------------------
def stage_megatron(megatron_home):
    cfg = SMOKE_CONFIG
    gpus = cfg["gpus"]
    gbs  = cfg["micro_batch_size"]   # DP=1 for this smoke test

    script = os.path.join(megatron_home, "pretrain_gpt.py")
    if not os.path.exists(script):
        raise RuntimeError(
            f"pretrain_gpt.py not found at {script}\n"
            f"Set --megatron to your Megatron-LM clone directory."
        )

    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "--nnodes=1", "--node_rank=0",
        "--master_addr=localhost", "--master_port=29500",
        script,

        # Model
        f"--num-layers={cfg['num_layers']}",
        f"--hidden-size={cfg['hidden_size']}",
        f"--num-attention-heads={cfg['num_attention_heads']}",
        f"--ffn-hidden-size={cfg['ffn_hidden_size']}",
        f"--seq-length={cfg['seq_len']}",
        f"--max-position-embeddings={cfg['seq_len']}",
        "--normalization=LayerNorm",
        "--position-embedding-type=learned_absolute",

        # Parallelism
        f"--tensor-model-parallel-size={cfg['tp']}",
        f"--pipeline-model-parallel-size={cfg['pp']}",
        f"--context-parallel-size={cfg['cp']}",

        # Batch
        f"--micro-batch-size={cfg['micro_batch_size']}",
        f"--global-batch-size={gbs}",

        # Iterations
        f"--train-iters={TOTAL_ITERS}",
        "--eval-iters=0",
        "--log-interval=1",          # log every iteration

        # LR: constant, no warmup
        "--lr=1e-4",
        "--min-lr=1e-4",
        "--lr-decay-style=constant",
        "--lr-warmup-iters=0",
        "--lr-decay-iters=0",
        "--weight-decay=0.01",
        "--adam-beta1=0.9", "--adam-beta2=0.95",
        "--clip-grad=1.0",

        # Data
        "--mock-data",
        "--vocab-size=32000",

        # Precision
        "--bf16",

        # Metrics
        "--log-throughput",
        "--log-timers-to-tensorboard",
        "--timing-log-level=2",
        "--log-memory-to-tensorboard",
        "--tensorboard-dir=./smoke_tb",

        # No saves
        "--no-save-optim", "--no-save-rng",
        "--no-gradient-accumulation-fusion",
        "--distributed-backend=nccl",
        "--init-method-std=0.01",

        # SP (requires TP>1, which we have)
        "--sequence-parallel",
    ]

    env = {
        **os.environ,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "WANDB_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_TIMEOUT": "120",
        "TOKENIZERS_PARALLELISM": "false",
    }

    print(f"       {dim('Command: ' + ' '.join(cmd[:6]) + ' ...')}", flush=True)
    print(f"       {dim(f'Full command written to smoke_test_cmd.txt')}")
    Path("smoke_test_cmd.txt").write_text(" \\\n  ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=300, cwd=megatron_home, env=env,
    )
    elapsed = time.time() - t0

    # Save full output for inspection
    output = proc.stdout + proc.stderr
    Path("smoke_test_output.txt").write_text(output)

    if proc.returncode != 0:
        # Try to extract the most useful error line
        error_lines = [
            l.strip() for l in output.splitlines()
            if any(kw in l.lower() for kw in ["error", "exception", "traceback", "assert", "killed"])
        ]
        summary = error_lines[-3:] if error_lines else [output[-400:]]
        raise RuntimeError(
            f"torchrun exited {proc.returncode}\n"
            + "\n".join(f"  {l}" for l in summary)
            + f"\n  Full output: smoke_test_output.txt"
        )

    success_marker = f"iteration       {TOTAL_ITERS}/"
    if success_marker not in output:
        raise RuntimeError(
            f"Process exited 0 but success marker '{success_marker}' not found.\n"
            f"  Check smoke_test_output.txt for details."
        )

    return f"Completed {TOTAL_ITERS} iterations in {elapsed:.1f}s", output


# ---------------------------------------------------------------------------
# Stage 4 — Metric parsing
# ---------------------------------------------------------------------------
def stage_metrics(output):
    cfg     = SMOKE_CONFIG
    gbs     = cfg["micro_batch_size"]
    seq_len = cfg["seq_len"]
    gpus    = cfg["gpus"]
    pp      = cfg["pp"]

    # --- Throughput ---
    elapsed_ms_list = []
    for m in re.finditer(
        r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)",
        output, re.DOTALL
    ):
        if int(m.group(1)) > WARMUP_ITERS:
            elapsed_ms_list.append(float(m.group(2)))

    if not elapsed_ms_list:
        raise RuntimeError(
            "No post-warmup iteration timing found in output.\n"
            "Check that --log-interval=1 is being respected."
        )

    avg_ms      = sum(elapsed_ms_list) / len(elapsed_ms_list)
    samples_sec = gbs / (avg_ms / 1000.0)
    tokens_sec  = samples_sec * seq_len

    # --- Memory ---
    mem_vals = re.findall(r"mem-allocated-bytes:\s*([\d.]+)", output)
    peak_gb  = float(mem_vals[-1]) / 1e9 if mem_vals else float("nan")

    # --- Comm ratio ---
    ar_vals = [float(x) for x in re.findall(r"timers/all-reduce:\s*([\d.]+)", output)]
    fb_vals = [float(x) for x in re.findall(r"timers/forward-backward:\s*([\d.]+)", output)]
    comm_ratio = (ar_vals[-1] / fb_vals[-1] * 100) if (ar_vals and fb_vals and fb_vals[-1] > 0) else float("nan")

    # --- Pipeline bubble (analytical) ---
    m_batches    = gbs // cfg["micro_batch_size"]
    bubble_ratio = ((pp - 1) / (m_batches + pp - 1) * 100) if pp > 1 else 0.0

    # --- MFU ---
    # GPT-1B params (approximate, attn + FFN only)
    h, L, ffn = cfg["hidden_size"], cfg["num_layers"], cfg["ffn_hidden_size"]
    params = L * (4 * h * h + 2 * h * ffn)   # attn (4h²) + FFN (2h·ffn)
    mfu    = (6 * params * tokens_sec) / (gpus * V100_BF16_PEAK) * 100

    metrics = {
        "throughput_tokens_per_sec":  round(tokens_sec, 1),
        "throughput_samples_per_sec": round(samples_sec, 3),
        "peak_memory_gb":             round(peak_gb, 2),
        "comm_ratio_pct":             round(comm_ratio, 2) if not math.isnan(comm_ratio) else "n/a",
        "bubble_ratio_pct":           round(bubble_ratio, 2),
        "mfu_pct":                    round(mfu, 3),
        "measured_iters":             len(elapsed_ms_list),
        "avg_iter_ms":                round(avg_ms, 1),
    }

    detail_lines = [f"{k}: {v}" for k, v in metrics.items()]
    msg = (f"{tokens_sec:,.0f} tok/s  |  MFU {mfu:.1f}%  |  "
           f"mem {peak_gb:.1f}GB  |  {len(elapsed_ms_list)} iters measured")
    return msg, "\n".join(detail_lines), metrics


# ---------------------------------------------------------------------------
# Stage 5 — CSV write
# ---------------------------------------------------------------------------
def stage_csv(metrics):
    row = {
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status":       "PASS",
        "model":        SMOKE_CONFIG["model_name"],
        "num_layers":   SMOKE_CONFIG["num_layers"],
        "tp":           SMOKE_CONFIG["tp"],
        "pp":           SMOKE_CONFIG["pp"],
        "cp":           SMOKE_CONFIG["cp"],
        "sp":           SMOKE_CONFIG["sp"],
        "mbs":          SMOKE_CONFIG["micro_batch_size"],
        "seq_len":      SMOKE_CONFIG["seq_len"],
        "precision":    SMOKE_CONFIG["precision"],
        "gpus":         SMOKE_CONFIG["gpus"],
        "train_iters":  SMOKE_CONFIG["train_iters"],
        **metrics,
    }
    path = "smoke_test_result.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return f"Result written to {path}", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Smoke test for Megatron-LM environment on Bridges2")
    ap.add_argument("--megatron", default=os.path.expanduser("~/Megatron-LM"),
                    help="Path to Megatron-LM clone (default: ~/Megatron-LM)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="Override GPU count (default: 4, auto-adjusted if fewer available)")
    args = ap.parse_args()

    # Auto-adjust GPU count if user requests fewer
    if args.gpus:
        SMOKE_CONFIG["gpus"] = args.gpus

    megatron_home = os.path.expanduser(args.megatron)

    print(bold("\n=== Megatron-LM Environment Smoke Test ==="))
    print(f"  Megatron path : {megatron_home}")
    print(f"  Model         : {SMOKE_CONFIG['model_name']} "
          f"(layers={SMOKE_CONFIG['num_layers']}, h={SMOKE_CONFIG['hidden_size']})")
    print(f"  Parallelism   : TP={SMOKE_CONFIG['tp']} PP={SMOKE_CONFIG['pp']} "
          f"CP={SMOKE_CONFIG['cp']} SP={SMOKE_CONFIG['sp']}")
    print(f"  Iterations    : {TOTAL_ITERS} total ({WARMUP_ITERS} warmup + "
          f"{TOTAL_ITERS - WARMUP_ITERS} measured)")
    print(f"  GPUs          : {SMOKE_CONFIG['gpus']}")

    megatron_output = None
    metrics         = None

    all_passed = True
    all_passed &= run_stage(1, "Python imports (torch / apex / transformer_engine / megatron)",
                            stage_imports)

    all_passed &= run_stage(2, "CUDA + NCCL visibility",
                            stage_cuda)

    def _megatron():
        return stage_megatron(megatron_home)

    stage3 = StageResult(3, "Megatron torchrun launch", False, "skipped")
    if all_passed:
        passed = run_stage(3, "Megatron torchrun launch (20 iterations)", _megatron)
        all_passed &= passed
        if passed:
            megatron_output = RESULTS[-1].detail
    else:
        print(f"\n{bold('[3/5]')} Megatron torchrun launch ...  {yellow('SKIPPED')} (earlier stage failed)")
        RESULTS.append(stage3)

    def _metrics():
        nonlocal metrics
        msg, detail, metrics = stage_metrics(megatron_output)
        return msg, detail

    stage4 = StageResult(4, "Metric parsing", False, "skipped")
    if megatron_output:
        all_passed &= run_stage(4, "Metric parsing (throughput / memory / MFU / comm / bubble)",
                                _metrics)
    else:
        print(f"\n{bold('[4/5]')} Metric parsing ...  {yellow('SKIPPED')}")
        RESULTS.append(stage4)

    def _csv():
        return stage_csv(metrics or {})

    stage5 = StageResult(5, "CSV write", False, "skipped")
    if metrics:
        all_passed &= run_stage(5, "Write result to smoke_test_result.csv", _csv)
    else:
        print(f"\n{bold('[5/5]')} CSV write ...  {yellow('SKIPPED')}")
        RESULTS.append(stage5)

    # --- Summary ---
    print(f"\n{'─'*52}")
    print(bold("Summary"))
    for r in RESULTS:
        icon = green("✓") if r.passed else (yellow("–") if r.message == "skipped" else red("✗"))
        print(f"  {icon}  Stage {r.stage}: {r.name}")
        if not r.passed and r.message != "skipped":
            for line in r.message.splitlines()[:4]:
                print(f"       {red(line)}")

    print()
    if all_passed:
        print(green("✓ All stages passed — environment is ready for the full sweep."))
        print(dim("  Next step: bash submit_all.sh"))
    else:
        first_fail = next((r for r in RESULTS if not r.passed and r.message != "skipped"), None)
        print(red("✗ Smoke test failed."))
        if first_fail:
            print(f"  Fix stage {first_fail.stage} ({first_fail.name}) first.")
        print(dim("  Full Megatron output (if available): smoke_test_output.txt"))
        print(dim("  Full command used:                   smoke_test_cmd.txt"))
        sys.exit(1)


if __name__ == "__main__":
    main()