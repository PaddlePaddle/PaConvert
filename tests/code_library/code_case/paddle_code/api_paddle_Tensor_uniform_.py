import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 3, 4, 9, 0.5, 1.5])
c = paddle.tensor(a.uniform_(min=2, max=6))
