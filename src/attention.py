import math
import torch

def scaled_dot_product_attention(Q,K,V,mask=None):
    dk = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(0,1))

    scores = scores / math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = torch.softmax(scores,dim=-1)
    output = torch.matmul(weights, V)

    return output,weights
