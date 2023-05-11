import paddle
print('#########################case1#########################')
a = paddle.to_tensor(data=[1, 3, 4, 9, 0.5, 1.5])
c = paddle.to_tensor(data=a.uniform_(min=2, max=6))
