from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LlamaModelGraph:
    model_name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    layers: list["LlamaLayerNode"]
    final_norm: "RMSNormNode"
    lm_head: "LMHeadNode"
    mode: str
    batch_size: int
    seq_len: int
    max_cache_len: int
    generation_mode: str


@dataclass
class LlamaLayerNode:
    layer_id: int
    input_norm: "RMSNormNode"
    attention: "AttentionDecodeNode"
    post_attention_norm: "RMSNormNode"
    mlp: "GatedMLPNode"


@dataclass
class AttentionDecodeNode:
    layer_id: int
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    uses_paged_kv_cache: bool


@dataclass
class GatedMLPNode:
    layer_id: int
    gate_proj: str
    up_proj: str
    down_proj: str
    activation: str


@dataclass
class RMSNormNode:
    weight: str
    eps: float


@dataclass
class LMHeadNode:
    weight: str
    tied_to_embedding: bool
    split_count: int | None


def graph_to_dict(graph: LlamaModelGraph) -> dict[str, Any]:
    return asdict(graph)


def graph_from_dict(data: dict[str, Any]) -> LlamaModelGraph:
    layers = [
        LlamaLayerNode(
            layer_id=layer["layer_id"],
            input_norm=RMSNormNode(**layer["input_norm"]),
            attention=AttentionDecodeNode(**layer["attention"]),
            post_attention_norm=RMSNormNode(
                **layer["post_attention_norm"]
            ),
            mlp=GatedMLPNode(**layer["mlp"]),
        )
        for layer in data["layers"]
    ]
    return LlamaModelGraph(
        model_name=data["model_name"],
        num_layers=data["num_layers"],
        hidden_size=data["hidden_size"],
        intermediate_size=data["intermediate_size"],
        num_attention_heads=data["num_attention_heads"],
        num_key_value_heads=data["num_key_value_heads"],
        head_dim=data["head_dim"],
        vocab_size=data["vocab_size"],
        rms_norm_eps=data["rms_norm_eps"],
        rope_theta=data["rope_theta"],
        layers=layers,
        final_norm=RMSNormNode(**data["final_norm"]),
        lm_head=LMHeadNode(**data["lm_head"]),
        mode=data["mode"],
        batch_size=data["batch_size"],
        seq_len=data["seq_len"],
        max_cache_len=data["max_cache_len"],
        generation_mode=data["generation_mode"],
    )
