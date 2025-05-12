import torch
import torch.nn as nn
from Embedding import PositionalEmbedding, WordEmbedding
from transformer_utils import tokenizer, mapping, get_vocab


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, dropout_prop):
        super().__init__()
        
        self.d_model = d_model
        
        self.pos_embedding = PositionalEmbedding(d_model, max_len, device)
        self.word_embedding = WordEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p= dropout_prop)
        
    def forward(self, x):
        word_emb = self.word_embedding(x)
        pos_emb = self.pos_embedding(x)
        
        return self.dropout((word_emb + pos_emb) * torch.sqrt(torch.tensor(self.d_model)))
    

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, vocab_size,  max_len, device, dropout_prop):
        super().__init__()
        
        self.emb = Embedding(vocab_size, d_model, max_len, device, dropout_prop)
        self.mul_attn = nn.MultiheadAttention(d_model, n_head, batch_first= True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)       
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        
        x = self.emb(x)
        
        attn_outputs, _ = self.mul_attn(x, x, x, need_weights= False)
        x = self.norm1(x + attn_outputs)
        
        ff_outputs = self.ff(x)
        x = self.norm2(x + ff_outputs)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_head, vocab_size,  max_len, device, dropout_prop):
        super().__init__()
        
        self.emb = Embedding(vocab_size, d_model, max_len, device, dropout_prop)
        self.masked_attn = nn.MultiheadAttention(d_model, n_head, batch_first= True)
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
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, 2048)
        self.relu = nn.ReLU(inplace= True)
        self.linear2 = nn.Linear(2048, d_model)
        
    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))
        return x
class Transformer(nn.Module):
    def __init__(self, d_model, n_head, num_encoder, num_decoder, vocab_size, max_len, dropout_prop, device):
        super().__init__()
    
        self.decoder_blocks= nn.ModuleList()
        self.encoder_blocks= nn.ModuleList()
        
        self.linear = nn.Linear(d_model, vocab_size)
        
        
        for _ in range(num_encoder):
            self.encoder_blocks.append(Encoder(d_model, n_head, vocab_size,  max_len, device, dropout_prop))
            
        for _ in range(num_decoder):
            self.decoder_blocks.append(Decoder(d_model, n_head, vocab_size,  max_len, device, dropout_prop))
            

    def forward(self, x_encode, x_decode):
        encoder_inputs = x_encode
        
        for encoder in self.encoder_blocks:      
            encoder_inputs = encoder(encoder_inputs)
        
        for decoder in self.decoder_blocks:
            x_decode = decoder(x_decode, encoder_inputs)
        
        x_decode = self.linear(x_decode)
        return x_decode
        

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