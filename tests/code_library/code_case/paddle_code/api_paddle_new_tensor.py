import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3])
b = paddle.tensor([4, 5, 6], dtype="float64", requires_grad=True)
print("#########################case2#########################")
a = paddle.tensor([1, 2, 3], dtype="int64")
b = paddle.tensor([4, 5, 6], dtype=a.dtype)
