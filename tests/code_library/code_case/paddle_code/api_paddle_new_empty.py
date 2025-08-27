import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_empty((3, 4), dtype="float64", requires_grad=True, pin_memory=True)
