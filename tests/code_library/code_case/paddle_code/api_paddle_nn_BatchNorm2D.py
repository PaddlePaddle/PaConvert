import paddle

print("#########################case1#########################")
bn = paddle.nn.BatchNorm2D(num_features=5)
print("#########################case2#########################")
bn = paddle.nn.BatchNorm2D(num_features=27)
print("#########################case3#########################")
paddle.nn.BatchNorm2D(
    num_features=10, epsilon=1e-05, weight_attr=False, bias_attr=False
)
