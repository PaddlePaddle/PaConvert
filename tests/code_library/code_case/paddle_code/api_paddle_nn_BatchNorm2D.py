import paddle
print('#########################case1#########################')
bn = paddle.nn.BatchNorm2D(num_features=5, use_global_stats=True,
    weight_attr=None, bias_attr=None)
print('#########################case2#########################')
bn = paddle.nn.BatchNorm2D(num_features=27, use_global_stats=True,
    weight_attr=None, bias_attr=None)
print('#########################case3#########################')
paddle.nn.BatchNorm2D(num_features=10, epsilon=1e-05, use_global_stats=True,
    weight_attr=None if False is None or False else False, bias_attr=None if
    False is None or False else False)
