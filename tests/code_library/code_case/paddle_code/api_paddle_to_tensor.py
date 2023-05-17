import paddle

print("#########################case1#########################")
a = paddle.to_tensor(
    data=paddle.to_tensor(data=[2, 3, 4]),
    dtype="float32",
    place=paddle.CUDAPinnedPlace(),
    stop_gradient=not True,
)
print("#########################case2#########################")
flag = True
a = paddle.to_tensor(
    data=paddle.to_tensor(data=[2, 3, 4]),
    dtype="float32",
    place=paddle.CUDAPinnedPlace(),
    stop_gradient=not flag,
)
