import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
a = paddle.tensor([1, 3, 4, 9, 0.5, 1.5])
c = paddle.tensor(a.uniform_(2, 6))
