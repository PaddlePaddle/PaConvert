import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_full([3, 4], 2.43, dtype=paddle.float64, requires_grad=True, pin_memory=True)
print("#########################case2#########################")
flag = False
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_full((2, 3), 4, requires_grad=flag)
