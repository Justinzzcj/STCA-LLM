import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.dim = dim
        self.heads = heads
        head_dim= dim // self.heads
        all_head_dim = head_dim * self.heads
        self.scale = self.dim ** -0.5
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim, all_head_dim*2, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        B, _, _ = x.shape
        q = self.q(x)
        kv = self.kv(y)
        kv = rearrange(kv, 'b n (e d) -> e b n d', e=2)
        k, v = kv[0], kv[1]
        q = q*self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.dropout(out)


class selfAttention(nn.Module):
   def __init__(self, dim, heads=16, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        head_dim= dim // self.heads
        all_head_dim = head_dim * self.heads
        self.scale = self.dim ** -0.5
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim, all_head_dim*2, bias=False)
        self.qkv = nn.Linear(dim, all_head_dim*3, bias=False)
        self.dropout = nn.Dropout(dropout)

   def forward(self, x):
        B, _, _ = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (e d) -> e b n d', e=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q*self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out