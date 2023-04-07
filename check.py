import torch

print(torch.cuda.is_available() and torch.version.hip)