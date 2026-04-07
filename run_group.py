#!/usr/bin/env python3
"""
run_group.py
------------
Runs all configs for one job group. Designed to be called from a SLURM
batch script once per group. Each call handles exactly the models in that
group and appends results to a group-specific CSV.

Usage:
    python run_group.py --group g1_dense_small [options]
    python run_group.py --group g3_moe_mixtral --input configs_v2.csv

Key changes from previous validate_configs.py:
  - 20 total iterations (10 warmup + 10 measured), not 2
  - --log-interval 1 so every iteration is captured
  - --lr-warmup-iters 0 for constant LR throughout
  - Parses stdout to extract per-metric averages over iters 11-20
  - Model arch flags (hidden-size, num-layers, etc.) come from model_groups.py
    rather than being hardcoded tiny dummies
  - Each group has its own output CSV and SLURM log dir
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

# Import group registry
sys.path.insert(0, os.path.dirname(__file__))
from model_groups import (
    ALL_GROUPS, ALL_MODELS, TOTAL_ITERS, WARMUP_ITERS, MEASURE_ITERS,
    ModelArch, JobGroup,
)

BRIDGES2_GPUS_PER_NODE = 8

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class Result:
    config_id:            str
    group_id:             str
    model_name:           str
    model_type:           str
    num_gpus:             int
    gpus_used:            int
    tp:                   int
    pp:                   int
    cp:                   int
    ep:                   int
    dp:                   int
    sp:                   str
    recompute:            str
    overlap_grad_reduce:  bool
    overlap_param_gather: bool
    overlap_tp_comm:      bool
    micro_batch_size:     int
    seq_len:              int
    precision:            str
    status:               str
    exit_code:            int
    duration_s:           float
    # Metric columns (NaN if run failed or warmup-only)
    throughput_tokens_per_sec: float
    throughput_samples_per_sec: float
    peak_memory_gb:       float
    comm_ratio_pct:       float
    bubble_ratio_pct:     float   # analytical only
    mfu_pct:              float   # derived post-run
    error_summary:        str
    command:              str

# ---------------------------------------------------------------------------
# Build Megatron command
# ---------------------------------------------------------------------------
def build_command(cfg: dict, arch: ModelArch, group: JobGroup,
                  megatron_home: str, gpus_used: int) -> list:
    tp  = int(cfg["tp"])
    pp  = int(cfg["pp"])
    cp  = tp
    ep  = int(cfg["ep"])
    mbs = int(cfg["micro_batch_size"])
    seq = int(cfg["seq_len"])
    sp  = cfg["sp"] == "on"
    rc  = cfg["recompute"]
    prec = cfg["precision"]

    model_parallel = tp * pp * cp
    dp = max(1, gpus_used // model_parallel)
    gbs = mbs * dp  # minimal global batch

    # num_layers must be divisible by pp
    num_layers = arch.num_layers
    # Round up to nearest multiple of pp if needed
    if num_layers % pp != 0:
        num_layers = ((num_layers // pp) + 1) * pp

    script = os.path.join(megatron_home, "pretrain_gpt.py")

    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus_used}",
        "--nnodes=1", "--node_rank=0",
        "--master_addr=localhost", "--master_port=29500",
        script,

        # ── Model architecture (real dims from model_groups.py) ──
        f"--num-layers={num_layers}",
        f"--hidden-size={arch.hidden_size}",
        f"--num-attention-heads={arch.num_attention_heads}",
        f"--ffn-hidden-size={arch.ffn_hidden_size}",
        f"--seq-length={seq}",
        f"--max-position-embeddings={seq}",
        f"--normalization={arch.normalization}",
        f"--position-embedding-type={arch.position_embedding}",

        # ── GQA ──
        f"--num-query-groups={arch.num_query_groups}",
        *( ["--group-query-attention"] if arch.num_query_groups != arch.num_attention_heads else [] ),

        # ── Activation / bias ──
        *( ["--swiglu"] if arch.swiglu else [] ),
        *( ["--untie-embeddings-and-output-weights"] if arch.untie_embeddings else [] ),
        *( ["--disable-bias-linear"] if arch.disable_bias else [] ),

        # ── Parallelism ──
        f"--tensor-model-parallel-size={tp}",
        f"--pipeline-model-parallel-size={pp}",
        f"--context-parallel-size={cp}",

        # ── Batch ──
        f"--micro-batch-size={mbs}",
        f"--global-batch-size={gbs}",

        # ── Iterations: 20 total, log every 1 ──
        f"--train-iters={TOTAL_ITERS}",
        "--eval-iters=0",
        "--log-interval=1",                  # CRITICAL: log every iteration

        # ── LR: constant, no warmup ──
        "--lr=1e-4",
        "--min-lr=1e-4",
        "--lr-decay-style=constant",
        "--lr-warmup-iters=0",               # no warmup ramp
        "--lr-decay-iters=0",
        "--weight-decay=0.01",
        "--adam-beta1=0.9", "--adam-beta2=0.95",
        "--clip-grad=1.0",

        # ── Data ──
        "--mock-data",
        "--vocab-size=32000",

        # ── Precision ──
        "--bf16" if prec == "bf16" else "--fp8-format=hybrid",

        # ── Metric logging ──
        "--log-throughput",
        "--log-timers-to-tensorboard",
        "--timing-log-level=2",
        "--log-memory-to-tensorboard",
        "--tensorboard-dir=./tb_logs",

        # ── No checkpoint saves ──
        "--no-save-optim", "--no-save-rng",

        # ── Stability ──
        "--no-gradient-accumulation-fusion",
        "--distributed-backend=nccl",
        "--init-method-std=0.01",
    ]

    # ── Sequence parallelism ──
    if sp:
        cmd.append("--sequence-parallel")

    # ── Recomputation ──
    if rc == "full":
        cmd += ["--recompute-granularity=full", "--recompute-method=uniform"]
    elif rc == "selective":
        cmd += ["--recompute-granularity=selective"]
    elif rc == "module_specific":
        cmd += ["--recompute-granularity=selective", "--recompute-num-layers=1"]

    # ── Overlap flags ──
    if cfg.get("overlap_grad_reduce") in (True, "True"):
        cmd.append("--overlap-grad-reduce")
    if cfg.get("overlap_param_gather") in (True, "True"):
        cmd.append("--overlap-param-gather")
    if cfg.get("overlap_tp_comm") in (True, "True"):
        cmd.append("--tp-comm-overlap")

    # ── MoE-specific ──
    if arch.is_moe:
        cmd += [
            f"--num-experts={arch.num_experts}",
            f"--expert-model-parallel-size={ep}",
            f"--moe-router-topk={arch.moe_router_topk}",
            # Use "none" balancing + random logits instead of aux_loss.
            # aux_loss routing is severely imbalanced for the first ~500 iters,
            # causing OOM and artificially low throughput in short runs.
            # --moe-apply-random-logits forces uniform token distribution
            # across experts from iteration 1, which is explicitly designed
            # for benchmarking (see Megatron-LM MoE README).
            # This makes our 20-iter throughput readings representative of
            # steady-state hardware performance rather than cold-start routing chaos.
            "--moe-router-load-balancing-type=none",
            "--moe-apply-random-logits",
            "--moe-token-dispatcher-type=alltoall",
        ]

    # ── Group-level extra flags (family-specific) ──
    cmd += group.extra_megatron_flags

    # ── FP8 extras (Hopper only — will fail on V100 but cleanly) ──
    if prec == "fp8":
        cmd += ["--fp8-amax-history-len=1", "--fp8-amax-compute-algo=max"]

    return cmd


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------
ITER_PATTERN = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?throughput:\s*([\d.]+)",
    re.DOTALL,
)
MEM_PATTERN  = re.compile(r"mem-allocated-bytes:\s*([\d.]+)")
TIMER_PATTERN = re.compile(r"timers/all-reduce:\s*([\d.]+)")
FB_PATTERN   = re.compile(r"timers/forward-backward:\s*([\d.]+)")

def parse_metrics(stdout: str, cfg: dict, arch: ModelArch, gpus_used: int) -> dict:
    """Extract averaged metrics from iterations WARMUP_ITERS+1 .. TOTAL_ITERS."""
    mbs   = int(cfg["micro_batch_size"])
    seq   = int(cfg["seq_len"])
    tp    = int(cfg["tp"])
    pp    = int(cfg["pp"])
    cp    = tp
    dp    = max(1, gpus_used // (tp * pp * cp))
    gbs   = mbs * dp

    # Collect per-iteration values
    elapsed_ms_list, tput_list = [], []
    for m in re.finditer(
        r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?throughput:\s*([\d.]+)",
        stdout, re.DOTALL
    ):
        it = int(m.group(1))
        if it > WARMUP_ITERS:
            elapsed_ms_list.append(float(m.group(2)))
            tput_list.append(float(m.group(3)))

    if not elapsed_ms_list:
        return {}

    avg_elapsed_s = (sum(elapsed_ms_list) / len(elapsed_ms_list)) / 1000.0
    samples_per_sec = gbs / avg_elapsed_s
    tokens_per_sec  = samples_per_sec * seq

    # Memory: last reported value
    mem_vals = MEM_PATTERN.findall(stdout)
    peak_mem_gb = float(mem_vals[-1]) / 1e9 if mem_vals else float("nan")

    # Comm ratio: (all-reduce time) / (forward-backward time)
    ar_vals = [float(x) for x in TIMER_PATTERN.findall(stdout)]
    fb_vals = [float(x) for x in FB_PATTERN.findall(stdout)]
    if ar_vals and fb_vals and fb_vals[-1] > 0:
        comm_ratio = (ar_vals[-1] / fb_vals[-1]) * 100.0
    else:
        comm_ratio = float("nan")

    # Bubble ratio: analytical 1F1B formula
    m_microbatches = gbs // mbs   # == dp (since gbs = mbs * dp)
    if pp > 1:
        bubble_ratio = ((pp - 1) / (m_microbatches + pp - 1)) * 100.0
    else:
        bubble_ratio = 0.0

    # MFU: 6 * params * tokens_per_sec / (gpus * peak_tflops)
    # V100 BF16 peak = 125 TFLOPS
    V100_PEAK_TFLOPS = 125e12
    params = _param_count(arch)
    if params and tokens_per_sec:
        mfu = (6 * params * tokens_per_sec) / (gpus_used * V100_PEAK_TFLOPS) * 100.0
    else:
        mfu = float("nan")

    return {
        "throughput_tokens_per_sec":  round(tokens_per_sec, 1),
        "throughput_samples_per_sec": round(samples_per_sec, 3),
        "peak_memory_gb":             round(peak_mem_gb, 3),
        "comm_ratio_pct":             round(comm_ratio, 2),
        "bubble_ratio_pct":           round(bubble_ratio, 2),
        "mfu_pct":                    round(mfu, 3),
    }


def _param_count(arch: ModelArch) -> Optional[int]:
    """Approximate parameter count from arch dims (used for MFU)."""
    # Standard transformer param count formula (excludes embedding)
    h = arch.hidden_size
    ffn = arch.ffn_hidden_size
    L = arch.num_layers
    heads = arch.num_attention_heads
    kv_heads = arch.num_query_groups
    # Attention: Q proj + K proj + V proj + O proj
    attn_params = L * (h * h + h * (h // heads) * kv_heads * 2 + h * h)
    # FFN: two or three matrices depending on SwiGLU
    ffn_factor = 3 if arch.swiglu else 2
    if arch.is_moe:
        # Only active experts contribute to active param count (approximate)
        active_experts = arch.moe_router_topk
        ffn_params = L * ffn_factor * h * ffn * active_experts
    else:
        ffn_params = L * ffn_factor * h * ffn
    return attn_params + ffn_params


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------
def cfg_id(cfg: dict) -> str:
    key = json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def run_one(cfg: dict, arch: ModelArch, group: JobGroup,
            megatron_home: str, timeout: int, dry_run: bool) -> Result:

    tp   = int(cfg["tp"])
    pp   = int(cfg["pp"])
    cp   = tp
    ep   = int(cfg["ep"])

    model_parallel = tp * pp * cp
    if model_parallel > BRIDGES2_GPUS_PER_NODE:
        dp = 0
        return _skip(cfg, arch, group, ep, dp, 0,
                     f"model_parallel={model_parallel} > {BRIDGES2_GPUS_PER_NODE}")

    max_dp   = BRIDGES2_GPUS_PER_NODE // model_parallel
    actual_dp = min(int(cfg["dp"]), max_dp)
    gpus_used = model_parallel * actual_dp

    cmd     = build_command(cfg, arch, group, megatron_home, gpus_used)
    cmd_str = " ".join(cmd)

    if dry_run:
        return _make_result(cfg, arch, group, ep, actual_dp, gpus_used,
                            "DRY_RUN", 0, 0.0, {}, "", cmd_str)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=megatron_home,
            env={**os.environ,
                 "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                 "WANDB_DISABLED": "true",
                 "NCCL_TIMEOUT": "120"},
        )
        elapsed = time.time() - t0
        output  = proc.stdout + proc.stderr

        if proc.returncode == 0 and f"iteration       {TOTAL_ITERS}/" in output:
            metrics = parse_metrics(output, cfg, arch, gpus_used)
            status  = "PASS"
            err     = ""
        elif "out of memory" in output.lower():
            metrics = {}; status = "OOM"; err = _first_err(output, "out of memory")
        elif proc.returncode != 0:
            metrics = {}; status = "FAIL"; err = _first_err(output, "Error") or output[-300:]
        else:
            metrics = {}; status = "ERROR"; err = "exit 0 but success marker missing"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        metrics = {}; status = "TIMEOUT"
        proc    = type("P", (), {"returncode": -1})()
        err     = f"killed after {timeout}s"
    except Exception as exc:
        elapsed = time.time() - t0
        metrics = {}; status = "ERROR"
        proc    = type("P", (), {"returncode": -2})()
        err     = str(exc)

    return _make_result(cfg, arch, group, ep, actual_dp, gpus_used,
                        status, proc.returncode, round(elapsed, 2),
                        metrics, err, cmd_str)


def _skip(cfg, arch, group, ep, dp, gpus, reason):
    return _make_result(cfg, arch, group, ep, dp, gpus,
                        "SKIP", -1, 0.0, {}, reason, "")


def _make_result(cfg, arch, group, ep, dp, gpus,
                 status, exit_code, duration, metrics, err, cmd_str):
    nan = float("nan")
    return Result(
        config_id=cfg_id(cfg),
        group_id=group.group_id,
        model_name=arch.name,
        model_type=arch.model_type,
        num_gpus=int(cfg["num_gpus"]),
        gpus_used=gpus,
        tp=int(cfg["tp"]), pp=int(cfg["pp"]),
        cp=int(cfg["tp"]),  # CP=TP always
        ep=ep, dp=dp,
        sp=cfg["sp"], recompute=cfg["recompute"],
        overlap_grad_reduce=cfg.get("overlap_grad_reduce", False),
        overlap_param_gather=cfg.get("overlap_param_gather", False),
        overlap_tp_comm=cfg.get("overlap_tp_comm", False),
        micro_batch_size=int(cfg["micro_batch_size"]),
        seq_len=int(cfg["seq_len"]),
        precision=cfg["precision"],
        status=status, exit_code=exit_code, duration_s=duration,
        throughput_tokens_per_sec=metrics.get("throughput_tokens_per_sec", nan),
        throughput_samples_per_sec=metrics.get("throughput_samples_per_sec", nan),
        peak_memory_gb=metrics.get("peak_memory_gb", nan),
        comm_ratio_pct=metrics.get("comm_ratio_pct", nan),
        bubble_ratio_pct=metrics.get("bubble_ratio_pct", nan),
        mfu_pct=metrics.get("mfu_pct", nan),
        error_summary=err[:300], command=cmd_str,
    )


def _first_err(text, kw):
    for line in text.splitlines():
        if kw.lower() in line.lower():
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# Completed set (for --resume)
# ---------------------------------------------------------------------------
def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add(row["config_id"])
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group",    required=True,
                    help="Group ID, e.g. g1_dense_small")
    ap.add_argument("--input",    default="configs_v2.csv")
    ap.add_argument("--outdir",   default="results",
                    help="Directory for per-group CSV outputs")
    ap.add_argument("--megatron", default=os.path.expanduser("~/Megatron-LM"))
    ap.add_argument("--workers",  type=int, default=1)
    ap.add_argument("--timeout",  type=int, default=300,
                    help="Seconds per config (20 iters needs more than validation)")
    ap.add_argument("--limit",    type=int, default=None)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    # Find the requested group
    group = next((g for g in ALL_GROUPS if g.group_id == args.group), None)
    if group is None:
        print(f"Unknown group '{args.group}'. Available: {[g.group_id for g in ALL_GROUPS]}")
        sys.exit(1)

    group_model_names = {m.name for m in group.models}

    # Load configs filtered to this group's models
    with open(args.input) as f:
        all_cfgs = list(csv.DictReader(f))
    configs = [c for c in all_cfgs if c["model_name"] in group_model_names]

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.group}_results.csv")

    if args.resume:
        done = load_done(out_path)
        before = len(configs)
        configs = [c for c in configs if cfg_id(c) not in done]
        print(f"Resume: skipping {before - len(configs):,} completed configs")

    if args.limit:
        configs = configs[:args.limit]

    print(f"\nGroup : {group.group_id} — {group.description}")
    print(f"Models: {[m.name for m in group.models]}")
    print(f"Configs: {len(configs):,}  |  Workers: {args.workers}  |  Timeout: {args.timeout}s")
    print(f"Iterations: {TOTAL_ITERS} total ({WARMUP_ITERS} warmup + {MEASURE_ITERS} measured)")
    print(f"Output: {out_path}\n")

    result_fields = [f.name for f in fields(Result)]
    write_header  = not (args.resume and os.path.exists(out_path))
    out_f  = open(out_path, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=result_fields)
    if write_header:
        writer.writeheader()

    counts = {}
    t_start = time.time()

    def process(cfg):
        arch = ALL_MODELS[cfg["model_name"]]
        return run_one(cfg, arch, group, args.megatron,
                       args.timeout, args.dry_run)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, c): (i, c) for i, c in enumerate(configs)}
        for fut in as_completed(futures):
            idx, cfg = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                # Belt-and-suspenders: run_one should never raise, but if it
                # somehow does (e.g. pickling error, unexpected signal), record
                # a synthetic FATAL row and keep going rather than crashing the
                # entire job group.
                nan = float("nan")
                arch = ALL_MODELS.get(cfg.get("model_name", ""), None)
                r = Result(
                    config_id=cfg_id(cfg),
                    group_id=group.group_id,
                    model_name=cfg.get("model_name", "unknown"),
                    model_type=cfg.get("model_type", "unknown"),
                    num_gpus=int(cfg.get("num_gpus", 0)),
                    gpus_used=0,
                    tp=int(cfg.get("tp", 0)), pp=int(cfg.get("pp", 0)),
                    cp=int(cfg.get("tp", 0)), ep=int(cfg.get("ep", 1)), dp=0,
                    sp=cfg.get("sp", ""), recompute=cfg.get("recompute", ""),
                    overlap_grad_reduce=False, overlap_param_gather=False,
                    overlap_tp_comm=False,
                    micro_batch_size=int(cfg.get("micro_batch_size", 0)),
                    seq_len=int(cfg.get("seq_len", 0)),
                    precision=cfg.get("precision", ""),
                    status="FATAL", exit_code=-99, duration_s=0.0,
                    throughput_tokens_per_sec=nan,
                    throughput_samples_per_sec=nan,
                    peak_memory_gb=nan,
                    comm_ratio_pct=nan,
                    bubble_ratio_pct=nan,
                    mfu_pct=nan,
                    error_summary=f"Unhandled exception in worker: {exc}"[:300],
                    command="",
                )

            counts[r.status] = counts.get(r.status, 0) + 1
            writer.writerow(asdict(r))
            out_f.flush()

            done_n = sum(counts.values())
            eta = ""
            if done_n > 1 and not args.dry_run:
                rate = done_n / (time.time() - t_start)
                eta  = f"  ETA {(len(configs)-done_n)/rate/60:.1f}m"

            status_str = r.status
            if r.status == "PASS":
                status_str += f"  tok/s={r.throughput_tokens_per_sec:.0f}  MFU={r.mfu_pct:.1f}%  mem={r.peak_memory_gb:.1f}GB"
            elif r.error_summary:
                status_str += f"  [{r.error_summary[:60]}]"

            print(f"  [{done_n:>6}/{len(configs)}] {status_str:<70}"
                  f"  TP={r.tp} PP={r.pp} EP={r.ep} SP={r.sp} {r.seq_len} {r.precision}{eta}")

    out_f.close()
    elapsed = time.time() - t_start
    print(f"\n{'='*56}")
    print(f"  Group {args.group} complete in {elapsed/60:.1f} min")
    for s, c in sorted(counts.items()):
        print(f"  {s:<10} {c:>6,}")
    print(f"  Output: {out_path}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
