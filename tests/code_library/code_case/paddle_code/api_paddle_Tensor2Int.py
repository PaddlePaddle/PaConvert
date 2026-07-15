import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")


def a(x: paddle.IntTensor):
    pass


print("#########################case2#########################")
a = paddle.IntTensor(2, 3, 6)
