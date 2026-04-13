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

# Load the model and Hyperparameters
checkpoint = torch.load('model.ckpt')
model = Model().to(device)
model.load_state_dict(state_dict=model.state_dict())
model.eval()

# load the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

start = 'A sales person should'
context = tokenizer.encode(start)
x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)

# run generation
with torch.no_grad():
    for _ in range(100):
        y = model.generate(x, max_new_tokens=100, temperature=1.0)
        print('--'*20)
        print(tokenizer.decode(y[0].tolist()))
        print('--'*20)

