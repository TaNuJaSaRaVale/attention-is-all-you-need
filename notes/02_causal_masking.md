# Phase 2 — Causal Masking

## Goal

Prevent future-token information from entering decoder self-attention.

## Attention Flow

```text
QKᵀ
 ↓
Scale
 ↓
Causal Mask
 ↓
Softmax
 ↓
Attention Weights
 ↓
× V
 ↓
Output



Causal masking makes training respect the same information constraint that exists during autoregressive generation: the model may use the past, but it cannot use the future.