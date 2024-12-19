import paddle

print("#########################case1#########################")
print(tuple([2, 8, 64, 64]))
print("#########################case2#########################")
assert tuple(paddle.randn(shape=[6, 5, 7]).shape) == tuple([6, 5, 7])
print("#########################case3#########################")
out = tuple([6, 5, 7])
shape_nchw = tuple([6, 5, 7])
assert out == tuple(shape_nchw)
print("#########################case4#########################")
print(tuple([1]))
print("#########################case5#########################")
shape = tuple([1])
