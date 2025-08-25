import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
out_0 = a.new_zeros(size=[3, 4]).astype("float64").pin_memory()
out_0.stop_gradient = not True
b = out_0
print("#########################case2#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_zeros(3, 4, requires_grad=True)
print("#########################case3#########################")
b = a.new_zeros(size=[3, 4])
