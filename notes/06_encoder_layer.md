# Phase 6 — Encoder Layer

## Goal

Combine Multi-Head Self-Attention and the Position-wise Feed-Forward Network into one Transformer encoder block.

## Architecture

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual + LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual + LayerNorm
  ↓
Output

```

Mathematical Form
$$ X_1 = LayerNorm(X + MultiHeadAttention(X)) $$ $$ X_2 = LayerNorm(X_1 + FFN(X_1)) $$
Residual Connection

Instead of:

$$ Output = Sublayer(X) $$

we use:

$$ Output = X + Sublayer(X) $$

This provides a direct path for information and gradients through the network.

Layer Normalization

LayerNorm normalizes the feature representation of each token.