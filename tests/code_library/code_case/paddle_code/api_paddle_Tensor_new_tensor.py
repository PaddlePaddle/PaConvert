import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 2, 3])
b = paddle.to_tensor(data=[4, 5, 6], dtype=paddle.float64, stop_gradient=not True)
print("#########################case2#########################")
a = paddle.tensor([1, 2, 3], dtype=paddle.int64)
b = paddle.to_tensor(data=[4, 5, 6], dtype=a.dtype)
