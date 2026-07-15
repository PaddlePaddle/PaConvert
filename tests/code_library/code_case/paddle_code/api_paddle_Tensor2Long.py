import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")


def a(x: paddle.LongTensor):
    pass


print("#########################case2#########################")
a = paddle.LongTensor(2, 3)
