import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointEmbedding(nn.Module):

    def __init__(self, embedding_dim, vocab_size, max_length, drop_prob):
        super(JointEmbedding, self).__init__()

        self.max_length = max_length
        self.drop_prob = drop_prob
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.token_emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.dropout = nn.Dropout(drop_prob)	
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, input_tensor):
        token_embedding = self.token_emb(input_tensor)
        token_embedding = self.norm(token_embedding)
        token_embedding = self.dropout(token_embedding)

        return token_embedding
