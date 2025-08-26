import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_zeros([3, 4], dtype="float64", requires_grad=True, pin_memory=True)
print("#########################case2#########################")
a = paddle.to_tensor(data=[1, 2, 3], dtype="int64")
b = a.new_zeros(3, 4, requires_grad=True)
print("#########################case3#########################")
b = a.new_zeros([3, 4])
