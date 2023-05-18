import paddle
print('#########################case1#########################')
a = paddle.randint(low=2, high=5, shape=[3, 4])
print('#########################case2#########################')
paddle.randint(low=0, high=10, shape=[2, 2])
print('#########################case3#########################')
a, b = 2, 25
a = paddle.randint(low=a, high=b, shape=[3, 4])
