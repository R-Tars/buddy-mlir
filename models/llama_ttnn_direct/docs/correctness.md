# Numerical Correctness

TTNN Direct treats functional execution and numerical correctness as separate
claims. A generated string or a structurally valid KV cache is not numerical
evidence.

The correctness artifact schema records:

- the model config digest, tokenizer, prompt digest, input token IDs, layer
  count, and reference dtype;
- the final prompt position after each selected decoder layer;
- the final normalized hidden state and complete LM-head logits;
- deterministic key/value cache coordinates across heads, prompt positions,
  and channels;
- SHA-256 digests of float32-normalized samples.

The comparison report computes PCC, cosine similarity, maximum and mean
absolute error, and RMSE. Top-token equality is reported independently because
a high logits PCC does not guarantee the same greedy token.

HF references use the first `N` model layers followed by the model's final norm
and LM-head. This matches TTNN Direct depth validation semantics for
`N = 1, 2, 4, 32`. Reference capture uses only effective prompt tokens; TTNN
prefill may retain right-side shape padding, but only the final valid prompt
position is compared.

The product `correctness` suite captures both sides and only passes when the
requested comparisons meet their gates. This path has not yet been validated
on P150A after the refactor; HF artifacts alone remain reference inputs, not a
passing correctness result.

Use `--hf-reference` to reuse a prior CPU artifact. The suite validates its
model config digest, prompt digest, depth, prefill length, dtype, and expected
`3 * layers + 2` checkpoint count before opening the TTNN runtime path.
