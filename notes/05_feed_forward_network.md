# Phase 5 — Position-wise Feed-Forward Network

## Goal

Transform each token representation independently after attention.

## Core Idea

```text
Input
  ↓
Linear
512 → 2048
  ↓
ReLU
  ↓
Linear
2048 → 512
  ↓
Output

```
Attention mixes information between tokens, while the position-wise FFN independently transforms the representation of each token.

```text
Why two linear layers?
→ Create a nonlinear transformation using ReLU between them.

Why 512 → 2048 → 512?
→ Expand into a larger feature space, transform, then project back.

Why "position-wise"?
→ The same FFN is independently applied to every token position.

Does FFN mix tokens?
→ No.

Does attention mix tokens?
→ Yes.

Why ReLU?
→ Introduces non-linearity.
```