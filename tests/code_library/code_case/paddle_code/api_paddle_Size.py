import paddle

print("#########################case1#########################")
print(paddle.Size([2, 8, 64, 64]))
print("#########################case2#########################")
assert paddle.randn(6, 5, 7).size() == paddle.Size([6, 5, 7])
print("#########################case3#########################")
out = paddle.Size([6, 5, 7])
shape_nchw = paddle.Size([6, 5, 7])
assert out == paddle.Size(shape_nchw)
print("#########################case4#########################")
print(paddle.Size([1]))
print("#########################case5#########################")
shape = paddle.Size([1])
