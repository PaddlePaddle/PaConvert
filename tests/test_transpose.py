import torch

a = torch.tensor(torch.tensor([2, 3, 4]), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True, pin_memory=False)
print('[tensor]: ', a.shape, a.dtype)


print('cuda.is_available: ', torch.cuda.is_available())


def a(x: torch.Tensor):
    pass

a = torch.Tensor(2, 3)
print('[Tensor]: ', a.shape, a.dtype)


def a(x: torch.LongTensor):
    pass

a = torch.LongTensor(2, 3)
print('[LongTensor]: ', a.shape, a.dtype)


def a(x: torch.IntTensor):
    pass

a = torch.IntTensor(2, 3, 6)
print('[IntTensor]: ', a.shape, a.dtype)


def a(x: torch.FloatTensor):
    pass

a = torch.FloatTensor(2, 3, 6)
print('[FloatTensor]: ', a.shape, a.dtype)


a = torch.nn.functional.interpolate(torch.randn(1, 2, 20, 20), [24, 24])
print('[nn.functional.interpolate]: ', a.shape)

a = torch.nn.functional.interpolate(torch.rand(1, 2, 20, 20), scale_factor=0.6)
print('[nn.functional.interpolate]: ', a.shape)


r = torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
print('[equal]: ', r)


a = torch.randint(2, 5, [3, 4], device=torch.device('cuda'))
print('[randint]: ', a.shape, a.min(), a.max())

torch.randint(10, [2, 2])
print('[randint]: ', a.shape, a.min(), a.max())


print(torch.__version__)

a = torch.tensor([1, 2, 3])
b = a.new_tensor([4, 5, 6], dtype=torch.float64)
print('[Tensor.new_tensor]: ', b)


b = torch.tensor(a.new_zeros([3, 4], dtype=torch.float64, requires_grad=True))
print('[Tensor.new_zeros]: ', b)

b = a.new_zeros([3, 4], dtype=torch.float64)
print('[Tensor.new_zeros]: ', b)


a = torch.tensor([1, 3, 4, 9, 0.5, 1.5])
c = torch.tensor(a.normal_(0.2, 0.3))
print('[Tensor.normal_]: ', c)


c = torch.tensor(a.uniform_(2, 6))
print('[Tensor.uniform_]: ', c)


x = torch.tensor([[1], [2], [3]])
y = x.expand(3, 4)
print('[Tensor.expand]: ', y.shape)


torch.random.manual_seed(23)


# # nonsupport
# torch.backends.cudnn.deterministic = True

# # nonsupport
# torch.backends.cudnn.benchmark = False