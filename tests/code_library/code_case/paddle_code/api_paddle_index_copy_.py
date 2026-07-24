import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
x = paddle.zeros(5, 3)
t = paddle.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float)
index = paddle.tensor([0, 4, 2])
x.index_copy_(0, index, t)
print("#########################case2#########################")
x = paddle.zeros(2, 1, 3, 3)
t = paddle.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype=paddle.float,
)
index = paddle.tensor([0, 1, 2])
x.index_copy_(2, index, t)
print("#########################case3#########################")
x = paddle.zeros(5, 3)
t = paddle.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float)
index = paddle.tensor([0, 4, 2])
y = x.index_copy_(0, index, t)
print("#########################case4#########################")
x = paddle.zeros(2, 1, 3, 3)
t = paddle.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype=paddle.float,
)
index = paddle.tensor([0, 1, 2])
y = x.index_copy_(2, index, t)
print("#########################case5#########################")
x = paddle.zeros(20)
t = paddle.tensor([1, 3, 4, 5], dtype=paddle.float)
index = paddle.tensor([0, 12, 2, 1])
y = x.index_copy_(0, index, t)
