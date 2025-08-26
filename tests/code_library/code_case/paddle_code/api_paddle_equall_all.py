import paddle

print("#########################case1#########################")
r = paddle.equal_all(
    x=paddle.tensor([1, 2]), y=paddle.tensor([1, 2])
).item()
