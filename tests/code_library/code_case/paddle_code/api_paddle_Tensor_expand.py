import paddle

print("#########################case1#########################")
x = paddle.tensor([[1], [2], [3]])
y = x.expand(3, 4)
print("#########################case2#########################")
x = paddle.tensor([[1], [2], [3]])
y = x.expand((3, 4))
