# Phase 4 — Positional Encoding

## Problem

Self-attention has no built-in notion of sequence order.

```text
"I love cats"
"cats love I"

Add positional information to token embeddings.

X=Embedding+PositionalEncoding
```

Original Transformer

The paper uses sinusoidal positional encoding.

$$ PE(pos,2i) = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right) $$ $$ PE(pos,2i+1) = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right) $$
Even dimensions → sin
Odd dimensions  → cos

Where is it added?
→ To the token embeddings before entering the encoder/decoder stack.