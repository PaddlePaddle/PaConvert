import paddle
print('#########################case1#########################')
x = paddle.rand(shape=[2, 3, 4, 4])
x.transpose(perm=[0, 2, 3, 1])
