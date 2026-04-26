# ===- ttir_llm.py ------------------------------------------------------------
#
# Buddy Graph → TTIR lowering for LLM-style ops (transformer / attention / KV).
# Merged into ``buddy.compiler.ops.ttir.ops_registry`` with CNN entries taking
# precedence on name collisions (see ``ttir.py``).
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

from typing import List

from .ttir import (
    TTIRSandbox,
    _bool_attr,
    _cast_to_sandbox_elt,
    _i32_attr,
    _reshape_to,
    _tensor_meta_shape_dtype,
    _ranked_type,
)
from ..graph.type import TensorDType


def _mlir_element_type_for_tensor_dtype(ctx, td, default_float_elt):
    """Same mapping as ``graph/ttir_import.py`` (kept local to avoid import cycles)."""
    from ttmlir.ir import BF16Type, F16Type, F32Type, IntegerType

    if td is None:
        return default_float_elt
    if isinstance(td, TensorDType):
        name = td.value
    else:
        name = str(td)
    if name in ("bfloat16", "bf16"):
        return BF16Type.get()
    if name in ("float16", "f16"):
        return F16Type.get()
    if name in ("float32", "f32"):
        return F32Type.get()
    if name in ("int64", "i64"):
        return IntegerType.get_signless(64, ctx)
    if name in ("int32", "i32"):
        return IntegerType.get_signless(32, ctx)
    if name in ("bool", "i1"):
        return IntegerType.get_signless(1, ctx)
    return default_float_elt


def _v(symbol_table, name):
    return symbol_table[(str(name), 0)]


def _scalar_promote(arg, peer, sb: TTIRSandbox):
    """Broadcast scalar Python values to ``peer``'s shape (element type)."""
    from ttmlir.dialects import ttir
    from ttmlir.ir import (
        DenseElementsAttr,
        FloatAttr,
        IntegerType,
        RankedTensorType,
        IntegerAttr,
    )

    sh = list(peer.type.shape)
    elt = peer.type.element_type
    rt_s = RankedTensorType.get(sh, elt)
    if isinstance(elt, IntegerType):
        attr = DenseElementsAttr.get_splat(
            rt_s, IntegerAttr.get(elt, int(arg))
        )
    else:
        attr = DenseElementsAttr.get_splat(
            rt_s, FloatAttr.get(elt, float(arg))
        )
    return ttir.constant(rt_s, attr, loc=sb.loc)


def _bin_operands(symbol_table, a0, a1, sb: TTIRSandbox):
    """Resolve two Buddy args to TTIR Values; supports one Python scalar + tensor."""

    def _get(arg, peer):
        key = (str(arg), 0)
        if key in symbol_table:
            return symbol_table[key]
        if hasattr(arg, "type"):
            return arg
        if peer is not None and isinstance(arg, (int, float, bool)):
            return _scalar_promote(arg, peer, sb)
        raise KeyError(key)

    try:
        left = _get(a0, None)
    except KeyError:
        left = None
    if left is not None:
        return left, _get(a1, left)
    right = _get(a1, None)
    return _get(a0, right), right


def _ranked_from_meta(node, sb: TTIRSandbox):
    shape, dt = _tensor_meta_shape_dtype(node)
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    from ttmlir.ir import RankedTensorType

    return RankedTensorType.get(list(shape), mel)


def _maybe_i32_index(v, sb: TTIRSandbox):
    """``ttir.update_cache`` examples use ``tensor<...xi32>`` indices."""
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    et = v.type.element_type
    if isinstance(et, IntegerType) and et.width == 64:
        sh = [int(x) for x in v.type.shape]
        i32 = IntegerType.get_signless(32, sb.ctx)
        rt = RankedTensorType.get(sh, i32)
        return ttir.typecast(rt, v, loc=sb.loc)
    return v


def flash_attention_for_cpu_prefill_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F32Type, RankedTensorType

    q = _v(symbol_table, node.args[0])
    k = _v(symbol_table, node.args[1])
    vval = _v(symbol_table, node.args[2])
    attn_mask = node.kwargs.get("attn_mask")
    scale = node.kwargs.get("scale")
    mask_v = None
    if attn_mask is not None:
        mask_v = _v(symbol_table, attn_mask)

    tm = node.tensor_meta
    if isinstance(tm, dict):
        out_sh = list(tm["shape"][0])
        lse_sh = list(tm["shape"][1]) if len(tm["shape"]) > 1 else [1]
        lse_dt = tm["dtype"][1] if isinstance(tm["dtype"], (list, tuple)) else "float32"
    else:
        out_sh = list(tm.shape)
        lse_sh = [1]
        lse_dt = TensorDType.Float32

    rt = RankedTensorType.get(out_sh, q.type.element_type)
    is_causal = mask_v is None
    scale_kw = float(scale) if scale is not None else None

    # --------------------------------------------------------------------
    # Workaround for tt-metal ``sdpa_flash_decode`` sfpi compile failure.
    #
    # When Q seq_len == 1 and a mask is provided (typical static-cache decode
    # step), ``tt-mlir``'s TTIRToTTNN converter promotes ttir.sdpa to
    # ttnn.scaled_dot_product_attention_decode, whose kernel
    # (``sdpa_flash_decode.cpp``) currently triggers the sfpi compiler bug:
    #   "cannot write sfpu vector to memory"
    # in ``_calculate_exponential_piecewise_.constprop.isra``.
    #
    # We pad Q's seq_len from 1 to TILE_HEIGHT (32) so TTIRToTTNN keeps the
    # standard ``scaled_dot_product_attention`` op instead. On a 32x32 tile
    # machine this adds essentially zero compute (both shapes occupy one
    # tile) but avoids the broken kernel entirely.
    # --------------------------------------------------------------------
    TILE = 32
    q_shape = list(q.type.shape)
    use_decode_workaround = (
        len(q_shape) == 4
        and int(q_shape[-2]) == 1
        and mask_v is not None
    )

    if use_decode_workaround:
        q_elt = q.type.element_type
        pad_q_shape = q_shape[:-2] + [TILE, int(q_shape[-1])]
        pad_q_ty = RankedTensorType.get(pad_q_shape, q_elt)
        bcast_q = [1, 1, TILE, 1]
        q_pad = ttir.broadcast(pad_q_ty, q, bcast_q, loc=sb.loc)

        m_shape = list(mask_v.type.shape)
        m_elt = mask_v.type.element_type
        pad_m_shape = m_shape[:-2] + [TILE, int(m_shape[-1])]
        pad_m_ty = RankedTensorType.get(pad_m_shape, m_elt)
        bcast_m = [1] * len(m_shape)
        bcast_m[-2] = TILE
        m_pad = ttir.broadcast(pad_m_ty, mask_v, bcast_m, loc=sb.loc)

        pad_out_shape = out_sh[:-2] + [TILE, int(out_sh[-1])]
        pad_out_ty = RankedTensorType.get(pad_out_shape, q_elt)
        out_pad = ttir.scaled_dot_product_attention(
            pad_out_ty,
            q_pad,
            k,
            vval,
            attention_mask=m_pad,
            is_causal=is_causal,
            scale=scale_kw,
            loc=sb.loc,
        )

        begins = [0] * len(pad_out_shape)
        ends = list(pad_out_shape)
        ends[-2] = 1
        step = [1] * len(pad_out_shape)
        out = ttir.slice_static(rt, out_pad, begins, ends, step, loc=sb.loc)
    else:
        out = ttir.scaled_dot_product_attention(
            rt,
            q,
            k,
            vval,
            attention_mask=mask_v,
            is_causal=is_causal,
            scale=scale_kw,
            loc=sb.loc,
        )
    lse_mel = _mlir_element_type_for_tensor_dtype(sb.ctx, lse_dt, F32Type.get())
    lse_ty = RankedTensorType.get(lse_sh, lse_mel)
    lse = ttir.empty(lse_ty, loc=sb.loc)
    return (out, lse)


def gqa_attention_fused_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    q = _v(symbol_table, node.args[0])
    k = _v(symbol_table, node.args[1])
    vval = _v(symbol_table, node.args[2])
    attn_mask = node.kwargs.get("attn_mask")
    scale = node.kwargs.get("scale")
    mask_v = None
    if attn_mask is not None:
        mask_v = _v(symbol_table, attn_mask)

    cur_key = (
        node.kwargs.get("cur_pos_tensor")
        or node.kwargs.get("cache_position")
        or node.kwargs.get("cur_pos")
    )
    if cur_key is None and len(node.args) > 3:
        cur_key = node.args[3]
    if cur_key is None:
        raise NotImplementedError(
            "GQAAttentionFusedOp → TTIR: set kwargs['cur_pos_tensor'] to the Buddy "
            "name of the cache-position placeholder (e.g. decode step index tensor) "
            "before lower_to_ttir()."
        )
    cur = _v(symbol_table, cur_key)

    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_type(out_shape, sb)
    scale_kw = float(scale) if scale is not None else None
    return ttir.scaled_dot_product_attention_decode(
        rt,
        q,
        k,
        vval,
        cur,
        attention_mask=mask_v,
        is_causal=False,
        scale=scale_kw,
        loc=sb.loc,
    )


def index_put_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    if len(node.args) < 3:
        raise NotImplementedError(f"IndexPutOp TTIR: expected >=3 args, got {node.args!r}")
    cache = _v(symbol_table, node.args[0])
    val = _v(symbol_table, node.args[2])
    spec = node.args[1]
    idx_names = [x for x in spec if x is not None]
    if len(idx_names) != 1:
        raise NotImplementedError(
            "IndexPutOp TTIR: only [None,…, index_tensor, …] with one index tensor "
            f"is supported; got {spec!r}"
        )
    update_index = _maybe_i32_index(_v(symbol_table, idx_names[0]), sb)
    out_ty = cache.type
    bo = _i32_attr(0, sb)
    return ttir.update_cache(out_ty, cache, val, update_index, batch_offset=bo, loc=sb.loc)


def embedding_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    weight = _v(symbol_table, node.args[0])
    indices = _v(symbol_table, node.args[1])
    rt = _ranked_from_meta(node, sb)
    return ttir.embedding(rt, indices, weight, loc=sb.loc)


def matmul_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _v(symbol_table, node.args[0])
    b = _v(symbol_table, node.args[1])
    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_type(out_shape, sb)
    return ttir.matmul(rt, a, b, loc=sb.loc)


def batch_matmul_op(node, symbol_table, sb: TTIRSandbox):
    return matmul_op(node, symbol_table, sb)


def mul_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.multiply(rt, a, b, loc=sb.loc)


def div_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.div(rt, a, b, loc=sb.loc)


def sub_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.subtract(rt, a, b, loc=sb.loc)


def rsub_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.subtract(rt, b, a, loc=sb.loc)


def silu_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.silu(rt, x, loc=sb.loc)


def gelu_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.gelu(rt, x, loc=sb.loc)


def pow_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.pow(rt, a, b, loc=sb.loc)


def rsqrt_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.rsqrt(rt, x, loc=sb.loc)


def sqrt_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.sqrt(rt, x, loc=sb.loc)


def _unary_ttir(op_name: str, node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return getattr(ttir, op_name)(rt, x, loc=sb.loc)


def cos_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("cos", node, symbol_table, sb)


def sin_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("sin", node, symbol_table, sb)


def tan_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("tan", node, symbol_table, sb)


def exp_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("exp", node, symbol_table, sb)


def log_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("log", node, symbol_table, sb)


def neg_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("neg", node, symbol_table, sb)


def mean_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    dims = node.args[1]
    keepdim = bool(node.args[2])
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    dim_list: List[int] = []
    rank = len(x.type.shape)
    for d in dims:
        di = int(d)
        if di < 0:
            di += rank
        dim_list.append(di)
    rt = _ranked_from_meta(node, sb)
    return ttir.mean(
        rt,
        x,
        _bool_attr(keepdim, sb),
        dim_arg=dim_list,
        loc=sb.loc,
    )


def softmax_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    dim = int(node.args[1])
    rt = _ranked_from_meta(node, sb)
    return ttir.softmax(rt, x, dim, numeric_stable=False, loc=sb.loc)


def concat_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    inputs_names = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    tensors = [_v(symbol_table, n) for n in inputs_names]

    def _numel(ty):
        n = 1
        for d in ty.shape:
            n *= int(d)
        return n

    # ``torch.cat`` can include empty tensors; TTIR rejects invalid rank/shape mixes.
    non_empty = [t for t in tensors if _numel(t.type) > 0]
    out_shape, _ = _tensor_meta_shape_dtype(node)
    out_shape = [int(s) for s in out_shape]
    if dim < 0:
        dim += len(out_shape)
    if not non_empty:
        raise NotImplementedError("concat TTIR: all operands are empty tensors.")
    if len(non_empty) == 1:
        only = non_empty[0]
        if [int(d) for d in only.type.shape] == out_shape:
            return only
        return _reshape_to(only, out_shape, node, sb)
    rt = _ranked_type(out_shape, sb)
    return ttir.concat(rt, non_empty, dim, loc=sb.loc)


def where_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    cond = _v(symbol_table, node.args[0])
    a = _v(symbol_table, node.args[1])
    b = _v(symbol_table, node.args[2])
    rt = _ranked_from_meta(node, sb)
    return ttir.where(rt, cond, a, b, loc=sb.loc)


def le_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _cast_to_sandbox_elt(_v(symbol_table, node.args[0]), sb)
    b = _cast_to_sandbox_elt(_v(symbol_table, node.args[1]), sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.le(rt, a, b, loc=sb.loc)


def lt_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _cast_to_sandbox_elt(_v(symbol_table, node.args[0]), sb)
    b = _cast_to_sandbox_elt(_v(symbol_table, node.args[1]), sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.lt(rt, a, b, loc=sb.loc)


def eq_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _cast_to_sandbox_elt(_v(symbol_table, node.args[0]), sb)
    b = _cast_to_sandbox_elt(_v(symbol_table, node.args[1]), sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.eq(rt, a, b, loc=sb.loc)


def ne_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _cast_to_sandbox_elt(_v(symbol_table, node.args[0]), sb)
    b = _cast_to_sandbox_elt(_v(symbol_table, node.args[1]), sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.ne(rt, a, b, loc=sb.loc)


def unsqueeze_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    x = _cast_to_sandbox_elt(x, sb)
    dim = int(node.args[1])
    out_shape, _ = _tensor_meta_shape_dtype(node)
    if dim < 0:
        dim += len(out_shape)
    rt = _ranked_type(out_shape, sb)
    return ttir.unsqueeze(rt, x, dim, loc=sb.loc)


def squeeze_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    x = _cast_to_sandbox_elt(x, sb)
    dim = int(node.args[1])
    rank = len(x.type.shape)
    if dim < 0:
        dim += rank
    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_type(out_shape, sb)
    return ttir.squeeze(rt, x, dim, loc=sb.loc)


def slice_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    x = _cast_to_sandbox_elt(x, sb)
    dim = int(node.args[1])
    start_idx = int(node.args[2])
    end_idx = int(node.args[3])
    sizes = [int(s) for s in x.type.shape]
    rank = len(sizes)
    if dim < 0:
        dim += rank
    if start_idx < 0:
        start_idx += sizes[dim]
    if end_idx < 0:
        end_idx += sizes[dim]
    begins = [0] * rank
    ends = list(sizes)
    step = [1] * rank
    begins[dim] = max(0, min(start_idx, sizes[dim]))
    ends[dim] = max(begins[dim], min(end_idx, sizes[dim]))
    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_type(out_shape, sb)
    return ttir.slice_static(rt, x, begins, ends, step, loc=sb.loc)


def expand_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    x = _cast_to_sandbox_elt(x, sb)
    out_shape, _ = _tensor_meta_shape_dtype(node)
    out_shape = [int(s) for s in out_shape]
    in_shape = [int(s) for s in x.type.shape]

    def _numel(sh):
        n = 1
        for d in sh:
            n *= int(d)
        return n

    # Pure rank / layout change with same elements (e.g. [8,8] → [1,8,8]).
    if _numel(in_shape) == _numel(out_shape) and tuple(in_shape) != tuple(
        out_shape
    ):
        return _reshape_to(x, out_shape, node, sb)

    rank_diff = len(out_shape) - len(in_shape)
    if rank_diff < 0:
        raise NotImplementedError(
            f"expand TTIR: output rank < input rank {in_shape} → {out_shape}"
        )
    in_padded = [1] * rank_diff + in_shape
    bcast = []
    for ins, outs in zip(in_padded, out_shape):
        if ins == outs:
            bcast.append(1)
        elif ins == 1:
            bcast.append(outs)
        else:
            raise NotImplementedError(
                f"expand TTIR: cannot broadcast {in_shape} → {out_shape}"
            )
    rt = _ranked_type(out_shape, sb)
    in_elt = x.type.element_type
    out_elt = rt.element_type
    # ttmlir-opt can assert when ``broadcast`` mixes dtypes (e.g. f32 in → bf16 out).
    # Broadcast in the input's element type, then typecast to the graph output type.
    if str(in_elt) != str(out_elt):
        from ttmlir.ir import RankedTensorType

        rt_same = RankedTensorType.get(list(out_shape), in_elt)
        expanded = ttir.broadcast(rt_same, x, bcast, loc=sb.loc)
        return ttir.typecast(rt, expanded, loc=sb.loc)
    return ttir.broadcast(rt, x, bcast, loc=sb.loc)


def lift_fresh_copy_op(node, symbol_table, sb: TTIRSandbox):
    """``lift_fresh_copy`` → same as identity-ish clone (multiply by one)."""
    return clone_op(node, symbol_table, sb)


def alias_op(node, symbol_table, sb: TTIRSandbox):
    """``aten.alias`` / tensor alias → identity path (multiply by 1)."""
    return clone_op(node, symbol_table, sb)


def clone_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    x = _v(symbol_table, node.args[0])
    rt = x.type
    elt = rt.element_type
    one = DenseElementsAttr.get_splat(rt, FloatAttr.get(elt, 1.0))
    ones_t = ttir.constant(rt, one, loc=sb.loc)
    return ttir.multiply(rt, x, ones_t, loc=sb.loc)


def convert_element_type_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.typecast(rt, x, loc=sb.loc)


def to_copy_op(node, symbol_table, sb: TTIRSandbox):
    """``aten._to_copy`` / dtype-device copies → ``ttir.typecast`` when dtype changes."""
    return convert_element_type_op(node, symbol_table, sb)


def tensor_constant_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import BF16Type, DenseElementsAttr

    import numpy as np
    import torch

    rt = _ranked_from_meta(node, sb)
    raw = node.args[0]
    arr = np.asarray(raw)
    elt = rt.element_type
    if isinstance(elt, BF16Type) and arr.dtype != np.uint16:
        # Frontend may supply float32 NumPy after folding bfloat16 tensors.
        t = torch.from_numpy(arr.astype(np.float32, copy=False)).to(torch.bfloat16)
        arr = t.view(torch.uint16).numpy()
    try:
        attr = DenseElementsAttr.get(arr, type=rt)
    except (TypeError, ValueError):
        attr = DenseElementsAttr.get(np.asarray(arr), type=rt)
    return ttir.constant(rt, attr, loc=sb.loc)


def full_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    fill = node.args[0]
    shape_arg = node.args[1]
    if isinstance(shape_arg, (list, tuple)):
        sh = [int(x) for x in shape_arg]
    else:
        sh = [int(shape_arg)]
    rt = _ranked_type(sh, sb)
    attr = DenseElementsAttr.get_splat(
        rt, FloatAttr.get(rt.element_type, float(fill))
    )
    return ttir.constant(rt, attr, loc=sb.loc)


def ones_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    sh = [int(x) for x in node.args[0]]
    rt = _ranked_type(sh, sb)
    return ttir.ones(rt, sh, loc=sb.loc)


def zeros_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    sh = [int(x) for x in node.args[0]]
    rt = _ranked_type(sh, sb)
    return ttir.zeros(rt, sh, loc=sb.loc)


def scalar_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    fill = node.args[0]
    rt = _ranked_from_meta(node, sb)
    attr = DenseElementsAttr.get_splat(
        rt, FloatAttr.get(rt.element_type, float(fill))
    )
    return ttir.constant(rt, attr, loc=sb.loc)


def iota_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    count = int(node.args[0])
    start = int(node.kwargs.get("start", 0))
    step = int(node.kwargs.get("step", 1))
    out_shape, dt = _tensor_meta_shape_dtype(node)
    if len(out_shape) != 1:
        raise NotImplementedError("IotaOp TTIR: only 1-D output is supported.")
    end = start + count * step
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    from ttmlir.ir import RankedTensorType

    rt = RankedTensorType.get(out_shape, mel)
    return ttir.arange(rt, start, end, step, 0, loc=sb.loc)


def arange_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    out_shape, dt = _tensor_meta_shape_dtype(node)
    if len(out_shape) != 1:
        raise NotImplementedError("ArangeOp TTIR: only 1-D is supported.")
    n = int(out_shape[0])
    start = int(node.kwargs.get("start", 0))
    step = int(node.kwargs.get("step", 1))
    if "end" in node.kwargs:
        end_excl = int(node.kwargs["end"])
    else:
        end_excl = start + n * step
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    rt = RankedTensorType.get(out_shape, mel)
    return ttir.arange(rt, start, end_excl, step, 0, loc=sb.loc)


def arange_start_step_op(node, symbol_table, sb: TTIRSandbox):
    return arange_op(node, symbol_table, sb)


def native_layer_norm_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F32Type, RankedTensorType

    x = _v(symbol_table, node.args[0])
    normalized_shape = [int(x) for x in node.args[1]]
    w = _v(symbol_table, node.args[2]) if node.args[2] is not None else None
    b = _v(symbol_table, node.args[3]) if node.args[3] is not None else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5
    rt = _ranked_from_meta(node, sb)
    y = ttir.layer_norm(
        rt,
        x,
        normalized_shape,
        weight=w,
        bias=b,
        epsilon=eps,
        loc=sb.loc,
    )
    tm = node.tensor_meta
    if (
        isinstance(tm, dict)
        and isinstance(tm.get("shape"), (list, tuple))
        and len(tm["shape"]) == 3
    ):
        shp = tm["shape"]
        dtp = tm["dtype"]
        msh = list(shp[1])
        rsh = list(shp[2])
        md0 = dtp[1] if isinstance(dtp, (list, tuple)) else TensorDType.Float32
        rd0 = dtp[2] if isinstance(dtp, (list, tuple)) else TensorDType.Float32
        mel_m = _mlir_element_type_for_tensor_dtype(sb.ctx, md0, F32Type.get())
        mel_r = _mlir_element_type_for_tensor_dtype(sb.ctx, rd0, F32Type.get())
        mean_e = ttir.empty(RankedTensorType.get(msh, mel_m), loc=sb.loc)
        rstd_e = ttir.empty(RankedTensorType.get(rsh, mel_r), loc=sb.loc)
        return (y, mean_e, rstd_e)
    return y


llm_ops_registry = {
    "FlashAttentionForCpuPrefillOp": flash_attention_for_cpu_prefill_op,
    "GQAAttentionFusedOp": gqa_attention_fused_op,
    "IndexPutOp": index_put_op,
    "EmbeddingOp": embedding_op,
    "MatmulOp": matmul_op,
    "BatchMatmulOp": batch_matmul_op,
    "MulOp": mul_op,
    "DivOp": div_op,
    "SubOp": sub_op,
    "RsubOp": rsub_op,
    "SiluOp": silu_op,
    "GeluOp": gelu_op,
    "PowOp": pow_op,
    "RsqrtOp": rsqrt_op,
    "SqrtOp": sqrt_op,
    "CosOp": cos_op,
    "SinOp": sin_op,
    "TanOp": tan_op,
    "ExpOp": exp_op,
    "LogOp": log_op,
    "NegOp": neg_op,
    "MeanOp": mean_op,
    "SoftmaxOp": softmax_op,
    "CatOp": concat_op,
    "WhereOp": where_op,
    "LeTensorOp": le_tensor_op,
    "LtTensorOp": lt_tensor_op,
    "EqTensorOp": eq_tensor_op,
    "NeTensorOp": ne_tensor_op,
    "UnsqueezeOp": unsqueeze_op,
    "SqueezeOp": squeeze_op,
    "SliceOp": slice_op,
    "ExpandOp": expand_op,
    "CloneOp": clone_op,
    "LiftFreshCopyOp": lift_fresh_copy_op,
    "AliasOp": alias_op,
    "ConvertElementTypeOp": convert_element_type_op,
    "ToCopyOp": to_copy_op,
    "TensorConstantOp": tensor_constant_op,
    "FullOp": full_op,
    "OnesOp": ones_op,
    "ZerosOp": zeros_op,
    "ScalarTensorOp": scalar_tensor_op,
    "IotaOp": iota_op,
    "ArangeOp": arange_op,
    "ArangeStartStepOp": arange_start_step_op,
    "NativeLayerNormOp": native_layer_norm_op,
}
