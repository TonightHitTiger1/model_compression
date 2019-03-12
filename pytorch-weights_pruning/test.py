import torch


# a = torch.Tensor(0.1,device='cuda:0')
a = torch.Tensor(0.1,device='cpu')
print(a)