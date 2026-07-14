import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3])
b = a.new_tensor([4, 5, 6], dtype=paddle.float64, requires_grad=True)
print("#########################case2#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = a.new_tensor([4, 5, 6])
