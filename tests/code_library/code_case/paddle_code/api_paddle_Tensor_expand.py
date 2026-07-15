import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
x = paddle.tensor([[1], [2], [3]])
y = x.expand(3, 4)
print("#########################case2#########################")
x = paddle.tensor([[1], [2], [3]])
y = x.expand((3, 4))
