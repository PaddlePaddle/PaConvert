import paddle

print("#########################case1#########################")
a = paddle.to_tensor(data=[1, 3, 4, 9, 0.5, 1.5])
x = a
a = paddle.assign(paddle.normal(mean=0.2, std=0.3, shape=x.shape).astype(x.dtype), x)
