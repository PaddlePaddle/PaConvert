import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_empty([3, 4], dtype=paddle.float64, requires_grad=True, pin_memory=True)
print("#########################case2#########################")
flag = False
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_empty((2, 3), requires_grad=flag)
