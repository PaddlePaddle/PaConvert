import paddle

print("#########################case1#########################")
x = paddle.zeros(shape=[5, 3])
t = paddle.to_tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
index = paddle.to_tensor(data=[0, 4, 2])
x.scatter_(index, t)
print("#########################case2#########################")
x = paddle.zeros(shape=[2, 1, 3, 3])
t = paddle.to_tensor(
    data=[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype="float32",
)
index = paddle.to_tensor(data=[0, 1, 2])
times, temp_shape, temp_index = (
    paddle.prod(paddle.to_tensor(x.shape[:2])),
    x.shape,
    index,
)
x, new_t = x.reshape([-1] + temp_shape[2 + 1 :]), t.reshape([-1] + temp_shape[2 + 1 :])
for i in range(1, times):
    temp_index = paddle.concat([temp_index, index + len(index) * i])
x.scatter_(temp_index, new_t).reshape(temp_shape)
print("#########################case3#########################")
x = paddle.zeros(shape=[5, 3])
t = paddle.to_tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
index = paddle.to_tensor(data=[0, 4, 2])
y = x.scatter_(index, t)
print("#########################case4#########################")
x = paddle.zeros(shape=[2, 1, 3, 3])
t = paddle.to_tensor(
    data=[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype="float32",
)
index = paddle.to_tensor(data=[0, 1, 2])
times, temp_shape, temp_index = (
    paddle.prod(paddle.to_tensor(x.shape[:2])),
    x.shape,
    index,
)
x, new_t = x.reshape([-1] + temp_shape[2 + 1 :]), t.reshape([-1] + temp_shape[2 + 1 :])
for i in range(1, times):
    temp_index = paddle.concat([temp_index, index + len(index) * i])
y = x.scatter_(temp_index, new_t).reshape(temp_shape)
print("#########################case5#########################")
x = paddle.zeros(shape=[20])
t = paddle.to_tensor(data=[1, 3, 4, 5], dtype="float32")
index = paddle.to_tensor(data=[0, 12, 2, 1])
y = x.scatter_(index, t)
