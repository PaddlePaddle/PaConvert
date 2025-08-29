import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3], dtype="int64")
out_0 = paddle.empty(shape=(3, 4), dtype="float64")
out_0.stop_gradient = not True
b = out_0.pin_memory()
