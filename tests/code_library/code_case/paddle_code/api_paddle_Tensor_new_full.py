import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
out_0 = paddle.full(shape=[3, 4], fill_value=2.43, dtype="float64")
out_0.stop_gradient = not True
b = out_0.pin_memory()
print("#########################case2#########################")
flag = False
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
out_1 = paddle.full(shape=(2, 3), fill_value=4, dtype=a.dtype)
out_1.stop_gradient = not flag
b = out_1
