# from numpy import dtype
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.Linear as Linear
# import torch.nn.functional as F


# a = torch.tensor([1, 2, 3])
# a.mean(0).max()


# torch.tensor([2, 3, 4], dtype=torch.float32, device=torch.device('cpu'), requires_grad=True, pin_memory=False)

# print(torch.cuda.is_available())

# a = torch.Tensor(2, 3, device=torch.device("cuda:0"))
# b = torch.LongTensor(2, 3, device=torch.device("cpu"))
# c = torch.IntTensor(2, 3, device=torch.device("cuda:0"))
# a = torch.FloatTensor(2, 3, device=torch.device("cuda:0"))

# a = torch.nn.functional.interpolate(x, [24, 24], scale_factor=0.6)

# torch.equal(x, y)

# torch.randint(2, 5, [3, 4], device=torch.device('cuda:1'))

# nonsupport
# print(torch.__version__)

# a = torch.tensor([1, 2, 3])

# a.new_zeros([3, 4]).half()

# b = a.new_tensor([1, 2, 3], dtype=torch.float32)

# b = torch.tensor(a.new_zeros([3, 4], dtype=torch.float64))

# c = torch.tensor(a.normal_(0.2, 0.3))

# c = torch.tensor([a.uniform_(2, 6)])

# c = a.expand(3, 4)

# torch.random.manual_seed(23)

# # nonsupport
# torch.backends.cudnn.deterministic = True

# # nonsupport
# torch.backends.cudnn.benchmark = False

# torch.tensor([1, 2, 3], dtype=torch.float64).new_zeros([1, 3])

# # # print('--------------')

torch.tensor([1, 2, 3], dtype=torch.float64, device=torch.device('cuda:0')).normal_(0.1, 0.3)

# torch.float16