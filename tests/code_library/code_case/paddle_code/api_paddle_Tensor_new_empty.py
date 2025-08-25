import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
out_0 = a.new_empty(size=[3, 4], dtype="float64").pin_memory()
out_0.stop_gradient = not True
b = out_0
print("#########################case2#########################")
flag = False
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
out_1 = a.new_empty(size=(2, 3))
out_1.stop_gradient = not flag
b = out_1
