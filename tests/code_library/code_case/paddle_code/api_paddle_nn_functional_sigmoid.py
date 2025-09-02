import paddle

print("#########################case1#########################")
t = paddle.randn(4)
paddle.nn.functional.sigmoid(t)
