# 内容参见mini-GPT.ipynb

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000 # 代表训练多少次迭代。每次迭代都会从训练数据中抽取一个batch进行训练
eval_interval = 300 # 每隔多少次迭代评估一次模型在训练集和验证集上的损失。
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 选择设备，如果有GPU可用则使用GPU，否则使用CPU
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # 将数据移动到设备上（GPU或CPU），以便后续的模型训练和评估能够在正确的设备上进行。
    return x, y

@torch.no_grad()
def estimate_loss(): # 评估模型在训练集和验证集上的损失。该函数在评估过程中不计算梯度，以节省内存和计算资源。函数会返回一个字典，包含训练集和验证集的平均损失。
    out = {}
    model.eval() # 将模型设置为评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters): # 评估eval_iters次，每次都从数据集中抽取一个batch进行评估，并计算损失。最后将所有评估的损失取平均，得到训练集和验证集的平均损失。
            X, Y = get_batch(split)  
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



# 开始训练

model = BigramLanguageModel(vocab_size)
m = model.to(device) # m是模型实例，model是模型类。我们将模型实例m移动到设备上（GPU或CPU），以便后续的训练和评估能够在正确的设备上进行。这步操作以后model和m指向同一个模型实例，只不过m在设备上，而model可能还在CPU上。我们后续的训练和评估都是通过m来进行的，因为m已经被移动到设备上了。

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb) # 为什么这里不是用m
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) 