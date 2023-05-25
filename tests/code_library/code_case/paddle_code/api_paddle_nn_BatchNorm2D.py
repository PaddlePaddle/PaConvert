import paddle
print('#########################case1#########################')
bn = paddle.nn.BatchNorm2D(num_features=5, momentum=1 - 0.1, epsilon=1e-05,
    weight_attr=None, bias_attr=None, use_global_stats=True)
print('#########################case2#########################')
bn = paddle.nn.BatchNorm2D(num_features=27, momentum=1 - 0.1, epsilon=1e-05,
    weight_attr=None, bias_attr=None, use_global_stats=True)
print('#########################case3#########################')
paddle.nn.BatchNorm2D(num_features=10, momentum=1 - 0.1, epsilon=1e-05,
    weight_attr=False, bias_attr=False, use_global_stats=True)
