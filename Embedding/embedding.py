import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device):
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sine/cosine positional embeddings")

        
        pe = torch.zeros(max_len, d_model, device= device)
        pe.requires_grad = False
        
        pos = torch.arange(0, max_len, dtype=torch.float, device= device).unsqueeze(1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        self.register_buffer("pe", pe)


    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return self.pe[:seq_len, :]

    
class WordEmbedding(nn.Embedding):
    def __init__(self,
                 vocab_size: int, 
                 d_model: int
                 ):
        
        super().__init__(vocab_size, d_model, padding_idx= 1)
        
        



