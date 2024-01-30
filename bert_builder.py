import torch

from torch import nn
import torch.nn.functional as f

class AttentionHead(nn.Module):

    def __init__(self, dim_inp, dim_out, drop_prob):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(drop_prob)
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.matmul(query, key.transpose(-1, -2)) / scale

        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)

        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out, drop_prob):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out,drop_prob) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)
    
class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads, dropout_prob):
        super(Encoder, self).__init__()
        self.dropout_prob = dropout_prob
        self.attention_heads = attention_heads
        self.dim_inp = dim_inp
        self.dim_out = dim_out
        
        
        self.attention = MultiHeadAttention(self.attention_heads, self.dim_inp, self.dim_out, self.dropout_prob)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(self.dim_inp, self.dim_out),
            nn.Dropout(self.dropout_prob),
            nn.GELU(),
            nn.Linear(self.dim_out, self.dim_inp),
            nn.Dropout(self.dropout_prob)
        )
        self.norm = nn.LayerNorm(self.dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)
