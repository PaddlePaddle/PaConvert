import paddle

print("#########################case1#########################")
a = paddle.tensor(
    paddle.tensor([2, 3, 4]),
    dtype=paddle.float32,
    device=paddle.device("cuda"),
    requires_grad=True,
    pin_memory=True,
)
print("#########################case2#########################")
flag = True
a = paddle.tensor(
    paddle.tensor([2, 3, 4]),
    dtype=paddle.float32,
    device=paddle.device("cuda"),
    requires_grad=flag,
    pin_memory=True,
)
print("#########################case3#########################")
a = paddle.tensor([2, 3, 4], device="cuda")
