import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(27, 10)
        self.fc1 = nn.Linear(30, 500)
        self.fc2 = nn.Linear(500, 27)

    def forward(self, x):
        out = self.embedding(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = self.fc2(out)
        return out


# read in all the words
words = open('name.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one?


def build_dataset(words_in):
    X, Y = [], []
    for w in words_in:

        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

mlp = MLP()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

lossi = []
stepi = []

for i in range(200000):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    parameters = list(mlp.parameters())

    # forward pass
    logits = mlp(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())

# plot loss
plt.plot(stepi, lossi)
plt.xlabel("Steps")
plt.ylabel("Log Loss")
plt.show()

logits = mlp(Xtr)
loss = F.cross_entropy(logits, Ytr)
print(f'tr: {loss}')

logits = mlp(Xdev)
loss = F.cross_entropy(logits, Ydev)
print(f'dev: {loss}')

logits = mlp(Xte)
loss = F.cross_entropy(logits, Yte)
print(f'te: {loss}')