from __future__ import annotations

from .graph import LlamaModelGraph


def validate_llama_graph(graph: LlamaModelGraph) -> None:
    errors: list[str] = []

    if graph.mode not in {"prefill", "decode"}:
        errors.append(f"unsupported mode: {graph.mode}")
    if graph.generation_mode not in {"greedy", "sampling"}:
        errors.append(
            f"unsupported generation_mode: {graph.generation_mode}"
        )
    if graph.num_layers != len(graph.layers):
        errors.append(
            "num_layers does not match layer count: "
            f"{graph.num_layers} != {len(graph.layers)}"
        )
    if graph.num_attention_heads <= 0:
        errors.append("num_attention_heads must be positive")
    if graph.num_key_value_heads <= 0:
        errors.append("num_key_value_heads must be positive")
    if graph.hidden_size <= 0:
        errors.append("hidden_size must be positive")
    if graph.head_dim <= 0:
        errors.append("head_dim must be positive")
    if graph.batch_size <= 0:
        errors.append("batch_size must be positive")
    if graph.seq_len <= 0:
        errors.append("seq_len must be positive")
    if graph.max_cache_len <= 0:
        errors.append("max_cache_len must be positive")

    expected_head_dim = graph.hidden_size // graph.num_attention_heads
    if graph.hidden_size % graph.num_attention_heads:
        errors.append("hidden_size must be divisible by num_attention_heads")
    elif graph.head_dim != expected_head_dim:
        errors.append(
            f"head_dim mismatch: {graph.head_dim} != {expected_head_dim}"
        )

    for expected_id, layer in enumerate(graph.layers):
        if layer.layer_id != expected_id:
            errors.append(
                f"layer id mismatch at index {expected_id}: "
                f"{layer.layer_id}"
            )
        if layer.attention.layer_id != layer.layer_id:
            errors.append(
                f"attention layer id mismatch in layer {layer.layer_id}"
            )
        if layer.mlp.layer_id != layer.layer_id:
            errors.append(f"mlp layer id mismatch in layer {layer.layer_id}")
        if layer.mlp.activation != "silu":
            errors.append(
                f"unsupported MLP activation in layer {layer.layer_id}: "
                f"{layer.mlp.activation}"
            )

    if errors:
        raise ValueError("invalid Llama semantic graph:\n- " + "\n- ".join(errors))
