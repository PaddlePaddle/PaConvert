import paddle
print('#########################case1#########################')
a = paddle.to_tensor(data=[1, 2, 3], dtype='int64')
out_0 = paddle.zeros(shape=[3, 4], dtype='float64')
out_0.stop_gradient = not True
b = out_0
print('#########################case2#########################')
a = paddle.to_tensor(data=[1, 2, 3], dtype='int64')
out_1 = paddle.zeros(shape=[3, 4], dtype=a.dtype)
out_1.stop_gradient = not True
b = out_1
print('#########################case3#########################')
b = paddle.zeros(shape=[3, 4], dtype=a.dtype)
