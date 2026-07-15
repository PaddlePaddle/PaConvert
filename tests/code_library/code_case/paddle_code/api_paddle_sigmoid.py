import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
t = paddle.randn(4)
paddle.sigmoid(t)
