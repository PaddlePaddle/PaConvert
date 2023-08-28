import paddle
print('#########################case1#########################')
paddle.nn.BatchNorm1D(num_features=3, epsilon=0.001, momentum=1 - 0.01)
print('#########################case2#########################')
bn = paddle.nn.BatchNorm1D(num_features=27)
print('#########################case3#########################')
paddle.nn.BatchNorm1D(num_features=10, epsilon=1e-05, weight_attr=None if 
    False else False, bias_attr=None if False else False)
