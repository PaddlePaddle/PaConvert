import paddle

print("#########################case1#########################")
a = paddle.nn.functional.interpolate(paddle.randn(1, 2, 20, 20), [24, 24])
print("#########################case2#########################")
a = paddle.nn.functional.interpolate(paddle.rand(1, 2, 20, 20), scale_factor=0.6)
