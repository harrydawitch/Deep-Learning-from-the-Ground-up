import torch
import torch.nn as nn
import numpy as np

from transformer_utils import tokenizer, mapping, get_vocab

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, 2048)
        self.relu = nn.ReLU(inplace= True)
        self.linear2 = nn.Linear(2048, d_model)
        
    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 ):
        super().__init__()
        
        self.mul_attn = nn.MultiheadAttention(d_model, n_head, batch_first= True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)       
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        
        attn_outputs, _ = self.mul_attn(x, x, x, need_weights= False)
        x = self.norm1(x + attn_outputs)
        
        ff_outputs = self.ff(x)
        x = self.norm2(x + ff_outputs)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                 d_model: int= 512,
                 n_head: int= 8
                 ):
        
        super().__init__()
        
        self.masked_attn = nn.MultiheadAttention(d_model, n_head, batch_first= True,)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_mul_attn = nn.MultiheadAttention(d_model, n_head, batch_first= True)
        self.norm2 = nn.LayerNorm(d_model)
                
        self.ff= FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_inputs):
        
        masked_attn_outputs, _ = self.masked_attn(x, x, x, is_causal= True)
        x = self.norm1(x + masked_attn_outputs)
        
        cross_attn_outputs, _ = self.cross_mul_attn(query= x, key= encoder_inputs, value= encoder_inputs)
        x = self.norm2(x + cross_attn_outputs)
        
        ff_outputs = self.ff(x)
        x = self.norm3(x + ff_outputs)
        
        return x       


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
           

class Transformer(nn.Module):
    def __init__(
                 self,
                 d_model: int= 512, 
                 n_head: int = 8,
                 num_encoder: int= 16, 
                 num_decoder: int= 16,
                 vocab_size: int= 10000
                 ):
        super().__init__()
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        
        self.pos_embedding = PositionalEmbedding(d_model, max_len= 5000)
        
        self.decoder_blocks= nn.ModuleList()
        self.encoder_blocks= nn.ModuleList()
        
        self.linear = nn.Linear(d_model, vocab_size)
        
        
        for _ in range(num_encoder):
            self.encoder_blocks.append(Encoder(d_model, n_head))
            
        for _ in range(num_decoder):
            self.decoder_blocks.append(Decoder(d_model, n_head))
            

    def forward(self, x_input, x_output):
        x_input = self.input_embedding(x_input)
        x_output = self.output_embedding(x_output)
        
        pos_embedding1 = self.pos_embedding(x_input)
        pos_embedding2 = self.pos_embedding(x_output)
        
        x_input += pos_embedding1
        x_output += pos_embedding2
       
        encoder_inputs = x_input
        
        for encoder in self.encoder_blocks:      
            encoder_inputs = encoder(encoder_inputs)
        
        for decoder in self.decoder_blocks:
            x_output = decoder(x_output, encoder_inputs)
            
        x_output = self.linear(x_output)
        
        return x_output
        

def test():
    DATA = r"Transformer\dataset.txt"
    D_MODEL = 512
    N_HEAD = 4
    NUM_ENCODER= 2
    NUM_DECODER= 2
    vocab_size = get_vocab(DATA)
    
    sentence = "Hi my name is Long, I am twenty one year old and i love to learn deep learning, AI"
    tokens = mapping(tokenizer(sentence), DATA, mode= "toidx")
    
    return tokens
    # model = Transformer(D_MODEL, N_HEAD, NUM_ENCODER, NUM_DECODER, vocab_size)
    
    # outputs = model(tokens)
    
if __name__ == "__main__":
    a = test()
    print(a)