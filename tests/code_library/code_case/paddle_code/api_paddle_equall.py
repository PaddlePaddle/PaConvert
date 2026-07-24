import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
r = paddle.equal(paddle.tensor([1, 2]), paddle.tensor([1, 2]))
