import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_zeros([3, 4], dtype=paddle.float64, requires_grad=True, pin_memory=True)
print("#########################case2#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_zeros(3, 4, requires_grad=True)
print("#########################case3#########################")
b = a.new_zeros([3, 4])
