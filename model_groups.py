"""
model_groups.py
---------------
Single source of truth for all model architecture definitions and job groupings.

Models are split into 6 SLURM job groups based on:
  1. Architecture family (dense GPT vs MoE variant)
  2. Megatron arg signature (different MoE families need different flags)
  3. Size tier (affects wall-time estimate per config)

Groups:
  group_1_dense_small   GPT-1B, GPT-7B
  group_2_dense_large   GPT-13B, GPT-30B
  group_3_moe_mixtral   Mixtral-8x7B-style, Mixtral-8x22B-style
  group_4_moe_deepseek  DeepSeekMoE-16B-style
  group_5_moe_qwen      Qwen3-30B-A3B-style
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Iteration / warmup constants (shared across all groups)
# ---------------------------------------------------------------------------
TOTAL_ITERS   = 20   # 10 warmup + 10 measured
WARMUP_ITERS  = 10   # discard first 10 in parser
MEASURE_ITERS = 10   # average iterations 11-20

# ---------------------------------------------------------------------------
# Model architecture dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelArch:
    name:              str
    model_type:        str          # "dense" | "moe"
    total_params:      str
    active_params:     str
    num_experts:       Optional[int]

    # Megatron transformer shape
    num_layers:        int
    hidden_size:       int
    ffn_hidden_size:   int
    num_attention_heads: int
    num_query_groups:  int          # GQA groups; equals num_heads for MHA

    # Architecture flags that differ by family
    normalization:     str          # "LayerNorm" | "RMSNorm"
    position_embedding: str         # "learned_absolute" | "rope"
    swiglu:            bool         # use SwiGLU activation
    untie_embeddings:  bool         # untie input/output embeddings
    disable_bias:      bool         # disable bias in linear layers

    # MoE-specific (None for dense)
    moe_router_topk:   Optional[int] = None
    # NOTE: moe_router_type is defined here for documentation but is overridden
    # in run_group.py to "none" + --moe-apply-random-logits for all short benchmark
    # runs. This forces balanced expert routing from iteration 1, avoiding the
    # ~500-iteration warmup penalty of aux_loss routing on fresh weights.
    moe_router_type:   str = "none"  # overridden at runtime

    # Rough per-config wall-time on V100-32 (seconds, for SLURM --time sizing)
    est_seconds_per_config: int = 120

    @property
    def is_moe(self):
        return self.model_type == "moe"

# ---------------------------------------------------------------------------
# Model definitions — real architecture parameters
# ---------------------------------------------------------------------------

# ── Dense GPT family ────────────────────────────────────────────────────────
# Uses standard learned positional embeddings and LayerNorm (GPT-2/GPT-3 style)
# num_query_groups == num_attention_heads (no GQA in original GPT-3 style)

GPT_1B = ModelArch(
    name="GPT-1B", model_type="dense",
    total_params="1B", active_params="1B", num_experts=None,
    num_layers=24, hidden_size=2048, ffn_hidden_size=8192,
    num_attention_heads=16, num_query_groups=16,
    normalization="LayerNorm", position_embedding="learned_absolute",
    swiglu=False, untie_embeddings=False, disable_bias=False,
    est_seconds_per_config=90,
)

GPT_7B = ModelArch(
    name="GPT-7B", model_type="dense",
    total_params="7B", active_params="7B", num_experts=None,
    num_layers=32, hidden_size=4096, ffn_hidden_size=16384,
    num_attention_heads=32, num_query_groups=32,
    normalization="LayerNorm", position_embedding="learned_absolute",
    swiglu=False, untie_embeddings=False, disable_bias=False,
    est_seconds_per_config=150,
)

GPT_13B = ModelArch(
    name="GPT-13B", model_type="dense",
    total_params="13B", active_params="13B", num_experts=None,
    num_layers=40, hidden_size=5120, ffn_hidden_size=20480,
    num_attention_heads=40, num_query_groups=40,
    normalization="LayerNorm", position_embedding="learned_absolute",
    swiglu=False, untie_embeddings=False, disable_bias=False,
    est_seconds_per_config=240,
)

GPT_30B = ModelArch(
    name="GPT-30B", model_type="dense",
    total_params="30B", active_params="30B", num_experts=None,
    num_layers=48, hidden_size=7168, ffn_hidden_size=28672,
    num_attention_heads=56, num_query_groups=56,
    normalization="LayerNorm", position_embedding="learned_absolute",
    swiglu=False, untie_embeddings=False, disable_bias=False,
    est_seconds_per_config=420,
)

# ── Mixtral family ───────────────────────────────────────────────────────────
# RMSNorm + RoPE + SwiGLU + GQA (8 KV heads) + 8 experts top-2
# Confirmed from Megatron-LM MoE README example script

MIXTRAL_8x7B = ModelArch(
    name="Mixtral-8x7B-style", model_type="moe",
    total_params="47B", active_params="13B", num_experts=8,
    num_layers=32, hidden_size=4096, ffn_hidden_size=14336,
    num_attention_heads=32, num_query_groups=8,
    normalization="RMSNorm", position_embedding="rope",
    swiglu=True, untie_embeddings=True, disable_bias=True,
    moe_router_topk=2, moe_router_type="aux_loss",
    est_seconds_per_config=210,
)

MIXTRAL_8x22B = ModelArch(
    name="Mixtral-8x22B-style", model_type="moe",
    total_params="141B", active_params="39B", num_experts=8,
    num_layers=56, hidden_size=6144, ffn_hidden_size=16384,
    num_attention_heads=48, num_query_groups=8,
    normalization="RMSNorm", position_embedding="rope",
    swiglu=True, untie_embeddings=True, disable_bias=True,
    moe_router_topk=2, moe_router_type="aux_loss",
    est_seconds_per_config=480,
)

# ── DeepSeekMoE family ───────────────────────────────────────────────────────
# Fine-grained experts (64 routed + 2 shared), top-6 routing, RMSNorm, RoPE
# Note: Megatron represents shared experts separately via --moe-shared-expert-intermediate-size

DEEPSEEK_MOE_16B = ModelArch(
    name="DeepSeekMoE-16B-style", model_type="moe",
    total_params="16B", active_params="2.8B", num_experts=64,
    num_layers=28, hidden_size=2048, ffn_hidden_size=1408,
    num_attention_heads=16, num_query_groups=16,
    normalization="RMSNorm", position_embedding="rope",
    swiglu=True, untie_embeddings=True, disable_bias=True,
    moe_router_topk=6, moe_router_type="aux_loss",
    est_seconds_per_config=300,
)

# ── Qwen3 MoE family ─────────────────────────────────────────────────────────
# 128 routed experts, top-8 routing, GQA (4 KV heads), RMSNorm, RoPE

QWEN3_30B_A3B = ModelArch(
    name="Qwen3-30B-A3B-style", model_type="moe",
    total_params="30B", active_params="3B", num_experts=128,
    num_layers=48, hidden_size=2048, ffn_hidden_size=768,
    num_attention_heads=32, num_query_groups=4,
    normalization="RMSNorm", position_embedding="rope",
    swiglu=True, untie_embeddings=True, disable_bias=True,
    moe_router_topk=8, moe_router_type="aux_loss",
    est_seconds_per_config=360,
)

# ---------------------------------------------------------------------------
# Job groups
# ---------------------------------------------------------------------------
@dataclass
class JobGroup:
    group_id:    str         # used as SLURM job name and output dir prefix
    description: str
    models:      list        # list of ModelArch
    # Estimated max wall-time for the whole group (hours) — set in SLURM script
    wall_hours:  float = 8.0
    # Additional Megatron flags that are family-specific (beyond the model shape)
    extra_megatron_flags: list = field(default_factory=list)

GROUP_DENSE_SMALL = JobGroup(
    group_id="g1_dense_small",
    description="Dense GPT: 1B and 7B",
    models=[GPT_1B, GPT_7B],
    wall_hours=6.0,
    extra_megatron_flags=[],
)

GROUP_DENSE_LARGE = JobGroup(
    group_id="g2_dense_large",
    description="Dense GPT: 13B and 30B",
    models=[GPT_13B, GPT_30B],
    wall_hours=8.0,
    extra_megatron_flags=[],
)

GROUP_MOE_MIXTRAL = JobGroup(
    group_id="g3_moe_mixtral",
    description="MoE Mixtral-style: 8x7B and 8x22B",
    models=[MIXTRAL_8x7B, MIXTRAL_8x22B],
    wall_hours=8.0,
    extra_megatron_flags=[
        "--no-position-embedding",          # RoPE replaces learned abs pos emb
        "--group-query-attention",          # GQA enabled
        "--no-masked-softmax-fusion",       # needed for MoE stability
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-token-dispatcher-type",      "alltoall",
    ],
)

GROUP_MOE_DEEPSEEK = JobGroup(
    group_id="g4_moe_deepseek",
    description="MoE DeepSeekMoE-16B style (64 fine-grained experts)",
    models=[DEEPSEEK_MOE_16B],
    wall_hours=6.0,
    extra_megatron_flags=[
        "--no-position-embedding",
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-token-dispatcher-type",      "alltoall",
        "--moe-shared-expert-intermediate-size", "1408",  # 2 shared experts
        "--moe-use-upcycling",              # helps stability at low iter counts
    ],
)

GROUP_MOE_QWEN = JobGroup(
    group_id="g5_moe_qwen",
    description="MoE Qwen3-30B-A3B style (128 routed experts)",
    models=[QWEN3_30B_A3B],
    wall_hours=8.0,
    extra_megatron_flags=[
        "--no-position-embedding",
        "--group-query-attention",
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-token-dispatcher-type",      "alltoall",
        "--no-masked-softmax-fusion",
    ],
)

ALL_GROUPS = [
    GROUP_DENSE_SMALL,
    GROUP_DENSE_LARGE,
    GROUP_MOE_MIXTRAL,
    GROUP_MOE_DEEPSEEK,
    GROUP_MOE_QWEN,
]

ALL_MODELS = {m.name: m for g in ALL_GROUPS for m in g.models}
