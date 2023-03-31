import torch
g_cpu = torch.Generator()
g_cuda = torch.Generator(device='cuda')
g_cuda = torch.Generator('cuda')
g_cuda = torch.Generator('cpu')