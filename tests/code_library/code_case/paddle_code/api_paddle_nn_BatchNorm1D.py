import paddle
print('#########################case1#########################')
paddle.nn.BatchNorm1D(num_features=3, momentum=1 - 0.01, epsilon=0.001,
    weight_attr=None, bias_attr=None, use_global_stats=True)
print('#########################case2#########################')
bn = paddle.nn.BatchNorm1D(num_features=27, momentum=1 - 0.1, epsilon=1e-05,
    weight_attr=None, bias_attr=None, use_global_stats=True)
print('#########################case3#########################')
paddle.nn.BatchNorm1D(num_features=10, momentum=1 - 0.1, epsilon=1e-05,
    weight_attr=False, bias_attr=False, use_global_stats=True)
