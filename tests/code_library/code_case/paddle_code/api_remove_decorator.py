import paddle

print("#########################case1#########################")


@paddle.jit.to_static
def a(x: paddle.IntTensor):
    pass
