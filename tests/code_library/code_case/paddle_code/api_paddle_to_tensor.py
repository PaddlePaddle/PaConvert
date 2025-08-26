import paddle

############################## 相关utils函数，如下 ##############################

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type
############################## 相关utils函数，如上 ##############################


print("#########################case1#########################")
a = paddle.tensor(
    paddle.tensor([2, 3, 4]),
    dtype="float32",
    place=device2str("cuda"),
    requires_grad=True,
    pin_memory=True,
)
print("#########################case2#########################")
flag = True
a = paddle.tensor(
    paddle.tensor([2, 3, 4]),
    dtype="float32",
    place=device2str("cuda"),
    requires_grad=flag,
    pin_memory=True,
)
print("#########################case3#########################")
a = paddle.tensor([2, 3, 4], device="cuda")
