import math
import torch

def scaled_dot_product_attention(Q,K,V):
    dk = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(0,1))

    scores = scores / math.sqrt(dk)

    weights = torch.softmax(scores,dim=-1)
    output = torch.matmul(weights, V)

    return output,weights
