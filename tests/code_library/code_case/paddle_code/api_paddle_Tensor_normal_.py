import paddle
print('#########################case1#########################')
a = paddle.to_tensor(data=[1, 3, 4, 9, 0.5, 1.5])
a = a.normal_(0.2, 0.3)
