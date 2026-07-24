import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
paddle.compat.nn.BatchNorm1d(3, eps=0.001, momentum=0.01)
print("#########################case2#########################")
bn = paddle.compat.nn.BatchNorm1d(27)
print("#########################case3#########################")
paddle.compat.nn.BatchNorm1d(10, eps=1e-05, affine=False)
