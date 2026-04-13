# Scaled Dot-Product Attention
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

Q = torch.rand(2,4)
K = torch.rand(2,4)
V = torch.rand(2,4)

out, w = attention(Q, K, V)
print("Output:", out)
print("Weights:", w)
