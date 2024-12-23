import paddle

print("#########################case1#########################")
a = paddle.nn.functional.interpolate(
    x=paddle.randn(shape=[1, 2, 20, 20]), size=[24, 24]
)
print("#########################case2#########################")
a = paddle.nn.functional.interpolate(
    x=paddle.rand(shape=[1, 2, 20, 20]), scale_factor=0.6
)
