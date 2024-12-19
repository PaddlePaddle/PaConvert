import paddle

print("#########################case1#########################")
t = paddle.randn(shape=[4])
paddle.nn.functional.sigmoid(x=t)
