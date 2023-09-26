import paddle
print('#########################case1#########################')
a = paddle.to_tensor(data=[1, 3, 4, 9, 0.5, 1.5])
"""Class Method: *.normal_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>a = a.normal_(0.2, 0.3)
