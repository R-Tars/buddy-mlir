from __future__ import annotations


_SOURCE = """\
GENERATED_NUM_LAYERS = __NUM_LAYERS__
GENERATED_LM_HEAD_SPLIT_COUNT = __LM_HEAD_SPLIT_COUNT__
GENERATED_RMS_NORM_EPS = __RMS_NORM_EPS__
"""


def render_constants(
    *,
    num_layers: int,
    lm_head_split_count: int,
    rms_norm_eps: float,
) -> str:
    return (
        _SOURCE.replace("__NUM_LAYERS__", str(num_layers))
        .replace("__LM_HEAD_SPLIT_COUNT__", str(lm_head_split_count))
        .replace("__RMS_NORM_EPS__", repr(rms_norm_eps))
    )
