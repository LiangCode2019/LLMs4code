import torch
import tiktoken
from model import Model

# Hyperparameters
batch_size = 12
context_length = 16
max_iters = 200
learning_rate = 1e-3
eval_interval = 20
eval_iters = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

with open('positive.txt', 'r') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('cl100k_base')
tokenizer_text = tokenizer.encode(text)
vocab_text = torch.tensor(tokenizer_text, dtype=torch.long).to(device)

p_size = int(len(tokenizer_text) * 0.9)
train_data = vocab_text[:p_size]
val_data = vocab_text[p_size:]

model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, hight=len(data)-context_length, size=(batch_size,)) # 12个随机数
    x = torch.stack([data[idx: idx+context_length] for idx in idxs]) #[batch_size, context_length]
    y = torch.stack([data[idx+1: idx+1+context_length] for idx in idxs])

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out #{'train': 2.345, 'val': 2.456}


for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    x, y = get_batch('train')
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), 'model.ckpt')