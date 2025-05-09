import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sine/cosine positional embeddings")

        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len]

    
class WordEmbedding(nn.Module):
    def __init__(self,
                 num_emb: int, 
                 emb_dim: int
                 ):
        
        super().__init__()
        
        self.embedding_matrix= nn.Parameter(torch.randn(num_emb, emb_dim))
        
    def forward(self, 
                x: list
                ):
        """Mapping each token inside list of tokens with their corresponding embedding vector

        Parameters
        ----------
        x : list
            list of tokens
        """
        return self.embedding_matrix[x, :]
        

