import paddle

print("#########################case1#########################")
bn = paddle.compat.nn.BatchNorm2d(5)
print("#########################case2#########################")
bn = paddle.compat.nn.BatchNorm2d(27)
print("#########################case3#########################")
paddle.compat.nn.BatchNorm2d(10, eps=1e-05, affine=False)
