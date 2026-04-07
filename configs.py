"""
generate_configs_v2.py
----------------------
Enumerates all valid parallelism configurations for distributed LLM/MoE training.

Validity rules applied:
  PARALLELISM
  1.  CP must equal TP
  2.  TP * PP * CP <= GPU count  (TP^2 * PP <= GPUs)
  3.  EP <= num_experts  (MoE only; dense locked to EP=1)
  4.  TP * PP * CP must divide GPU count evenly (DP = GPUs / (TP*PP*CP) >= 1)
  5.  DP (data parallel degree) must be >= 1 (implied by rule 4)
  6.  For MoE: EP must also divide GPU count cleanly alongside TP*PP*CP
      i.e. GPUs must be divisible by TP * PP * CP * EP ... or at minimum EP <= DP
      (EP runs within the DP dimension, so EP <= DP)

  SEQUENCE PARALLELISM
  7.  SP=on requires TP > 1 (SP is a TP-linked optimisation, meaningless at TP=1)

  PIPELINE
  8.  PP=2 requires at least 2 micro-batches in flight (micro_batch_size >= 1 always
      true, but global batch must allow it; we enforce micro_batch_size >= PP as a
      proxy so the pipeline is never starved)

  MEMORY (heuristic guards — prevents obviously OOM configs)
  9.  Large seq_len with small TP is risky for big models; we flag:
        - 30B+ dense models: seq_len=8192 requires TP >= 2
        - 141B MoE (Mixtral-8x22B): seq_len >= 4096 requires TP >= 2
  10. Full recomputation with micro_batch_size=8 and seq_len=8192 is always valid
      (recompute trades compute for memory, so it actually *helps* — no restriction here)

  PRECISION
  11. FP8 requires a Hopper/Blackwell GPU (H100, GH200, RTX5000).
      A100 does not support FP8 — but since we are not sweeping GPU platform in this
      script (platform is a separate hardware axis), we keep FP8 for all counts and
      note that A100 jobs must be filtered to BF16 downstream.
      → We include a `fp8_requires_hopper` flag column instead of hard-dropping rows.

  OVERLAP FLAGS
  12. TP comm overlap requires TP > 1 (no TP comm to overlap at TP=1)
  13. Param gather overlap is only meaningful when DP > 1
  14. Gradient reduce overlap is only meaningful when DP > 1

  MICRO BATCH / GLOBAL BATCH
  15. micro_batch_size must be <= global_batch / DP. We don't fix global batch size,
      so instead we enforce micro_batch_size >= 1 (always true) and note that the
      caller must set GBS >= micro_batch_size * DP.

Output columns per row:
  model_name, model_type, total_params, active_params, num_experts,
  num_gpus, tp, pp, cp, ep, dp,
  sp, recompute,
  overlap_grad_reduce, overlap_param_gather, overlap_tp_comm,
  micro_batch_size, seq_len, precision,
  fp8_requires_hopper,          ← informational flag
  rule_notes                    ← any soft warnings
"""

import csv
import itertools
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
TP_OPTIONS          = [1, 2, 4]
PP_OPTIONS          = [1, 2]
# CP derived: CP = TP
EP_OPTIONS          = [1, 2, 4, 8]       # MoE only
SP_OPTIONS          = ["on", "off"]
RECOMPUTE_OPTIONS   = ["none", "selective", "full", "module_specific"]
OVERLAP_COMBOS      = list(itertools.product([True, False], repeat=3))
MICRO_BATCH_OPTIONS = [1, 2, 4, 8]
SEQ_LEN_OPTIONS     = [2048, 4096, 8192]
PRECISION_OPTIONS   = ["bf16", "fp8"]
GPU_OPTIONS         = [8, 16, 32, 64]

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
@dataclass
class Model:
    name:         str
    type:         str             # "dense" or "moe"
    total_params: str
    active_params: str
    num_experts:  Optional[int]   # None for dense
    param_scale:  float           # rough relative scale for memory heuristics (B)

MODELS = [
    Model("GPT-1B",              "dense", "1B",   "1B",    None,  1),
    Model("GPT-7B",              "dense", "7B",   "7B",    None,  7),
    Model("GPT-13B",             "dense", "13B",  "13B",   None,  13),
    Model("GPT-30B",             "dense", "30B",  "30B",   None,  30),
    Model("Mixtral-8x7B-style",  "moe",   "47B",  "13B",   8,     47),
    Model("Mixtral-8x22B-style", "moe",   "141B", "39B",   8,     141),
    Model("DeepSeekMoE-16B",     "moe",   "16B",  "2.8B",  64,    16),
    Model("Qwen3-30B-A3B-style", "moe",   "30B",  "3B",    128,   30),
]

# ---------------------------------------------------------------------------
# Config row
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model_name:           str
    model_type:           str
    total_params:         str
    active_params:        str
    num_experts:          str
    num_gpus:             int
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
    fp8_requires_hopper:  bool
    rule_notes:           str = ""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(model: Model, num_gpus, tp, pp, cp, ep,
             sp, recompute, grad_red, param_gath, tp_comm,
             mbs, seq_len, precision):
    """
    Returns (is_valid, dp, fp8_flag, notes) or (False, ...) to skip.
    """
    notes = []

    # --- Rule 1: CP = TP (enforced by construction) ---
    assert cp == tp

    # --- Rule 2: TP * PP * CP <= GPU count ---
    model_parallel = tp * pp * cp
    if model_parallel > num_gpus:
        return False, 0, False, ""

    # --- Rule 4: GPU count divisible by model_parallel (DP must be integer) ---
    if num_gpus % model_parallel != 0:
        return False, 0, False, ""

    dp = num_gpus // model_parallel

    # --- Rule 5: DP >= 1 (guaranteed by rule 4, but be explicit) ---
    assert dp >= 1

    # --- Rule 3 + 6: EP constraints for MoE ---
    if model.type == "dense":
        if ep != 1:
            return False, 0, False, ""
    else:
        if ep > model.num_experts:
            return False, 0, False, ""
        # Rule 6: EP <= DP (EP runs within the DP dimension)
        if ep > dp:
            return False, 0, False, ""

    # --- Rule 7: SP=on requires TP > 1 ---
    if sp == "on" and tp == 1:
        return False, 0, False, ""

    # --- Rule 8: PP=2 requires micro_batch_size >= PP (pipeline not starved) ---
    if pp == 2 and mbs < pp:
        return False, 0, False, ""

    # --- Rule 12: TP comm overlap requires TP > 1 ---
    if tp_comm and tp == 1:
        return False, 0, False, ""

    # --- Rule 13: Param gather overlap requires DP > 1 ---
    if param_gath and dp == 1:
        return False, 0, False, ""

    # --- Rule 14: Gradient reduce overlap requires DP > 1 ---
    if grad_red and dp == 1:
        return False, 0, False, ""

    # --- Rule 9: Memory heuristics for large models + large seq_len ---
    # 30B+ dense: seq_len=8192 requires TP >= 2
    if model.type == "dense" and model.param_scale >= 30 and seq_len == 8192 and tp < 2:
        return False, 0, False, ""
    # Mixtral-8x22B (141B total): seq_len >= 4096 requires TP >= 2
    if model.param_scale >= 141 and seq_len >= 4096 and tp < 2:
        return False, 0, False, ""

    # --- Rule 11: FP8 flag (informational, not a hard filter) ---
    fp8_flag = (precision == "fp8")
    if fp8_flag:
        notes.append("fp8_requires_hopper_or_blackwell_gpu")

    return True, dp, fp8_flag, "|".join(notes)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def generate_configs():
    rows = []

    for model in MODELS:
        for num_gpus in GPU_OPTIONS:
            for tp in TP_OPTIONS:
                cp = tp  # Rule 1
                for pp in PP_OPTIONS:
                    ep_range = EP_OPTIONS if model.type == "moe" else [1]
                    for ep in ep_range:
                        for sp in SP_OPTIONS:
                            for recompute in RECOMPUTE_OPTIONS:
                                for (grad_red, param_gath, tp_comm) in OVERLAP_COMBOS:
                                    for mbs in MICRO_BATCH_OPTIONS:
                                        for seq_len in SEQ_LEN_OPTIONS:
                                            for precision in PRECISION_OPTIONS:

                                                ok, dp, fp8_flag, notes = validate(
                                                    model, num_gpus,
                                                    tp, pp, cp, ep,
                                                    sp, recompute,
                                                    grad_red, param_gath, tp_comm,
                                                    mbs, seq_len, precision
                                                )
                                                if not ok:
                                                    continue

                                                rows.append(Config(
                                                    model_name=model.name,
                                                    model_type=model.type,
                                                    total_params=model.total_params,
                                                    active_params=model.active_params,
                                                    num_experts=str(model.num_experts) if model.num_experts else "N/A",
                                                    num_gpus=num_gpus,
                                                    tp=tp, pp=pp, cp=cp, ep=ep, dp=dp,
                                                    sp=sp,
                                                    recompute=recompute,
                                                    overlap_grad_reduce=grad_red,
                                                    overlap_param_gather=param_gath,
                                                    overlap_tp_comm=tp_comm,
                                                    micro_batch_size=mbs,
                                                    seq_len=seq_len,
                                                    precision=precision,
                                                    fp8_requires_hopper=fp8_flag,
                                                    rule_notes=notes,
                                                ))
    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
def write_csv(rows, path="configs_v2.csv"):
    if not rows:
        print("No valid configurations found.")
        return
    fieldnames = [f.name for f in fields(Config)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWritten {len(rows):,} configurations to {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(rows):
    from collections import Counter

    total_before = sum(
        len(TP_OPTIONS) * len(PP_OPTIONS) * (len(EP_OPTIONS) if m.type == "moe" else 1)
        * len(SP_OPTIONS) * len(RECOMPUTE_OPTIONS) * len(OVERLAP_COMBOS)
        * len(MICRO_BATCH_OPTIONS) * len(SEQ_LEN_OPTIONS) * len(PRECISION_OPTIONS)
        for m in MODELS
        for _ in GPU_OPTIONS
    )

    print(f"\n{'='*56}")
    print(f"  Configurations before filtering : {total_before:>10,}")
    print(f"  Valid configurations            : {len(rows):>10,}")
    print(f"  Reduction                       : {100*(1 - len(rows)/total_before):>9.1f}%")
    print(f"{'='*56}")

    by_model = Counter(r.model_name for r in rows)
    print("\nBy model:")
    for name, count in by_model.most_common():
        print(f"  {name:<28} {count:>8,}")

    by_gpu = Counter(r.num_gpus for r in rows)
    print("\nBy GPU count:")
    for gpus in sorted(by_gpu):
        print(f"  {gpus:>2} GPUs {by_gpu[gpus]:>10,}")

    by_type = Counter(r.model_type for r in rows)
    print("\nBy model type:")
    for mtype, count in by_type.most_common():
        print(f"  {mtype:<10} {count:>10,}")

    by_prec = Counter(r.precision for r in rows)
    print("\nBy precision:")
    for prec, count in by_prec.most_common():
        print(f"  {prec:<8} {count:>10,}")

    by_recompute = Counter(r.recompute for r in rows)
    print("\nBy recompute strategy:")
    for rc, count in by_recompute.most_common():
        print(f"  {rc:<20} {count:>8,}")

    rules_applied = {
        " 1": "CP = TP (by construction)",
        " 2": "TP * PP * CP <= GPU count",
        " 3": "EP <= num_experts (MoE only)",
        " 4": "GPU count divisible by TP*PP*CP (integer DP)",
        " 5": "DP >= 1",
        " 6": "EP <= DP (EP runs within DP dimension)",
        " 7": "SP=on requires TP > 1",
        " 8": "PP=2 requires micro_batch_size >= PP",
        " 9": "Memory heuristic: large model + large seq_len requires TP >= 2",
        "10": "FP8 flagged as requiring Hopper/Blackwell (informational)",
        "11": "TP comm overlap requires TP > 1",
        "12": "Param gather overlap requires DP > 1",
        "13": "Gradient reduce overlap requires DP > 1",
    }
    print("\nRules applied:")
    for k, v in rules_applied.items():
        print(f"  [{k}] {v}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = generate_configs()
    print_summary(rows)
    write_csv(rows, "configs_v2.csv")