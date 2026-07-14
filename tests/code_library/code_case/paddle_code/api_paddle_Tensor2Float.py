import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")


def a(x: paddle.FloatTensor):
    pass


print("#########################case2#########################")
a = paddle.FloatTensor(2, 3, 6)
