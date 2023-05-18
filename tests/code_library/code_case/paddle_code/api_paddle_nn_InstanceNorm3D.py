import paddle
print('#########################case1#########################')
m = paddle.nn.InstanceNorm3D(num_features=100, momentum=1 - 0.1, epsilon=
    1e-05, weight_attr=False, bias_attr=False)
input = paddle.randn(shape=[20, 100, 35, 45, 10])
output = m(input)
print('#########################case2#########################')
m = paddle.nn.InstanceNorm3D(num_features=100, momentum=1 - 0.1, epsilon=
    1e-05, weight_attr=None, bias_attr=None)
input = paddle.randn(shape=[20, 100, 35, 45, 10])
output = m(input)
print('#########################case3#########################')
m = paddle.nn.InstanceNorm3D(num_features=100, momentum=1 - 0.1, epsilon=
    1e-05, weight_attr=False, bias_attr=False)
input = paddle.randn(shape=[20, 100, 35, 45, 10])
output = m(input)
print('#########################case4#########################')
m = paddle.nn.InstanceNorm3D(num_features=100, momentum=1 - 0.1, epsilon=
    1e-05, weight_attr=None, bias_attr=None)
input = paddle.randn(shape=[20, 100, 35, 45, 10])
output = m(input)
print('#########################case5#########################')
m = paddle.nn.InstanceNorm3D(num_features=100, momentum=1 - 0.1, epsilon=
    1e-05, weight_attr=False, bias_attr=False)
input = paddle.randn(shape=[20, 100, 35, 45, 10])
output = m(input)
