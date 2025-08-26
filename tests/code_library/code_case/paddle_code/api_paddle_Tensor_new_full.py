import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_full([3, 4], 2.43, dtype="float64", requires_grad=True, pin_memory=True)
print("#########################case2#########################")
flag = False
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_full((2, 3), 4, requires_grad=flag)
