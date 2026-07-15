import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
bn = paddle.nn.BatchNorm2d(5)
print("#########################case2#########################")
bn = paddle.nn.BatchNorm2d(27)
print("#########################case3#########################")
paddle.nn.BatchNorm2d(10, eps=1e-05, affine=False)
