import paddle

print("#########################case1#########################")
r = paddle.equal_all(
    x=paddle.to_tensor(data=[1, 2]), y=paddle.to_tensor(data=[1, 2])
).item()
