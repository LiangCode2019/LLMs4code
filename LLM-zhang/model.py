import torch
import torch.nn as nn
import math
from torch.nn import functional as F

# Hyperparameters
batch_size = 4
d_model = 512
context_length = 16
num_heads = 8
head_size = d_model // num_heads
num_blocks = 12
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForwadNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.ffn(x)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size, bias=False)
        self.Wk = nn.Linear(d_model, head_size, bias=False)
        self.Wv = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length)))
        self.Dropout = nn.Dropout(dropout)
   
    def forward(self,x):
        # x:[batch_size, [Timestep] context_length, head_size]
        B, T, D = x.shape
        
        q = self.Wq(x) # x:[batch_size, context_length, head_size]
        k = self.Wk(x)
        v = self.Wv(x)

        output = (q @ k.transpose(-2,-1)) / math.sqrt(head_size)
        output = output.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        output = F.softmax(output, dim=-1)
        output = self.Dropout(output)
        output = output @ v
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.Dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        output = torch.cat([head(x) for head in self.heads], dim=1)
        output = self.Dropout(self.Wo(output))
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwadNetwork()
    
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

class Model(nn.Module):
    def __init__(self, max_token_value=1000256):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model, max_token_value)
        self.te_lookup_table = nn.Embedding(max_token_value, d_model) # token embedding
        self.transformer_block = nn.Sequential(
            *([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)])
        )
    
    def forward(self,x_batch, y_batch=None):
        # x_batch:[batch_size, [Timestep] context_length, head_size]
        B, T, D = x_batch.shape
        pe_lookup_tabel = torch.zeros(context_length, d_model, device=device) # [context_length, d_model]
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0)* torch.arange(0, d_model, 2).float() / d_model) 
        pe_lookup_tabel[:, 0::2] = torch.sin(position*div_term)
        pe_lookup_tabel[:, 1::2] = torch.cos(position*div_term)

        output = self.te_lookup_table(x_batch) + pe_lookup_tabel
        output = self.transformer_block(output)
        logits = self.vocab_linear(output)

        if y_batch is not None:
            B, T, D = logits.shape
            logits_reshape = logits.view(B*T, D)
            y_batch_reshape = y_batch.view(B*T)
            loss = F.cross_entropy(logits_reshape,y_batch_reshape)
        else:
            loss = None
        
        return logits, loss
    
    # x_batch: [batch_size = 1, context_length = 16, (d_model=512)]
    def generate(self, x_batch, max_new_tokens=100, temperature=1.0,top_k=None):
        for _ in range(max_new_tokens):
            # x_batch [batch_size, context_length(Timestep)]
            x_crop = x_batch[:, -context_length:] # context_length
            logits, _ = self.forward(x_crop) #[batch_size, context_length(timrstep), vacab_size(100256)]
            logits = logits[:, -1, :] / temperature # [1, 1, vacab_size(100256)]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # token
            x_batch = torch.cat((x_batch, next_token), dim=1) # [batch_size, context_length+1]
        return x_batch
    