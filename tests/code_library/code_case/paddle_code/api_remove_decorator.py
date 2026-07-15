import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")


@paddle.jit.to_static
def a(x: paddle.IntTensor):
    pass
