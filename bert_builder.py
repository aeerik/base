import torch

from torch import nn
import torch.nn.functional as f
from embedding import JointEmbedding

class AttentionHead(nn.Module):

    def __init__(self, dim_embedding, drop_prob):
        super(AttentionHead, self).__init__()

        self.dim_embedding = dim_embedding
        self.dropout = nn.Dropout(drop_prob)
        self.q = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.k = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.v = nn.Linear(self.dim_embedding, self.dim_embedding)

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

    def __init__(self, num_heads, dim_embedding, drop_prob):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_embedding,drop_prob) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_embedding * num_heads, dim_embedding)
        self.norm = nn.LayerNorm(dim_embedding)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)

class resEncoder(nn.Module):

    def __init__(self, dim_embedding, attention_heads, dropout_prob):
        super(resEncoder, self).__init__()
        self.dropout_prob = dropout_prob
        self.attention_heads = attention_heads
        self.dim_embedding = dim_embedding
        self.dense = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.layer_norm = nn.LayerNorm(self.dim_embedding)
        self.dropout = nn.Dropout(self.dropout_prob)       
        
        self.attention = MultiHeadAttention(self.attention_heads, self.dim_embedding, self.dropout_prob)  
        self.feed_forward = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_embedding),
            nn.GELU(),
            nn.Linear(self.dim_embedding, self.dim_embedding),
            nn.Dropout(self.dropout_prob)
        )
        self.norm = nn.LayerNorm(self.dim_embedding)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        x = input_tensor
        context = self.attention(input_tensor, attention_mask)

        hidden_states = self.dense(context)
        hidden_states = self.dropout(hidden_states)
        x = x + hidden_states
        x = self.layer_norm(x)
        
        res = x
        x = self.feed_forward(x)
        x = x + res
        x = self.layer_norm(x)
        return x

class BERT_pt(nn.Module):

    def __init__(self, vocab_size, max_length, dim_embedding, attention_heads, num_encoders, dropout_prob):
        super(BERT_pt, self).__init__()
        self.attention_heads = attention_heads
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.dim_embedding = dim_embedding
        self.num_encoders = num_encoders
        self.dropout_prob = dropout_prob  

        self.embedding = JointEmbedding(self.dim_embedding, self.vocab_size, self.max_length, self.dropout_prob)
        self.encoders = nn.ModuleList([resEncoder(self.dim_embedding, self.attention_heads, self.dropout_prob) for _ in range(self.num_encoders)])

        self.token_prediction_layer = nn.Linear(self.dim_embedding, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        for layer in self.encoders:
            embedded = layer(embedded, attention_mask)

        token_predictions = self.token_prediction_layer(embedded)

        return self.softmax(token_predictions)    

class BERT_ft(nn.Module):

    def __init__(self, vocab_size, max_length, dim_embedding, dim_hidden, attention_heads, num_encoders, dropout_prob, num_ab, device):
        super(BERT_ft, self).__init__()
        self.attention_heads = attention_heads
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden    
        self.num_encoders = num_encoders
        self.dropout_prob = dropout_prob  

        self.embedding = JointEmbedding(self.dim_embedding, self.vocab_size, self.max_length, self.dropout_prob)
        self.encoders = nn.ModuleList([resEncoder(self.dim_embedding, self.attention_heads, self.dropout_prob) for _ in range(self.num_encoders)])
        self.token_prediction_layer = nn.Linear(self.dim_embedding, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.BC = [BC_Ab(self.dim_embedding, self.dim_hidden).to(device) for _ in range(num_ab)]

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        for layer in self.encoders:
            embedded = layer(embedded, attention_mask)
        
        cls_tokens = embedded[:, 0, :]
        resistance_predictions = torch.cat([network(cls_tokens) for network in self.BC], dim=1)
        token_predictions = self.token_prediction_layer(embedded)
        token_predictions = self.softmax(token_predictions)

        return token_predictions, resistance_predictions 

class BC_Ab(nn.Module): 
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(BC_Ab, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1), # binary classification (S:0 | R:1)
        )
           
    def forward(self, X):
        # X is the CLS token of the BERT model
        return self.classifier(X)