import paddle

print("#########################case1#########################")
print(list([2, 8, 64, 64]))
print("#########################case2#########################")
assert paddle.randn(shape=[6, 5, 7]).shape == list([6, 5, 7])
print("#########################case3#########################")
out = list([6, 5, 7])
shape_nchw = list([6, 5, 7])
assert out == list(shape_nchw)
print("#########################case4#########################")
print(list([1]))
print("#########################case5#########################")
shape = list([1])
