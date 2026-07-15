import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
x = paddle.rand([2, 3, 4, 4])
x.permute(0, 2, 3, 1)
