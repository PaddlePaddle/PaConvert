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
cpu = device2str("cpu")
a = paddle.randn(shape=[2, 3])
c = paddle.randn(shape=[2, 3], dtype=paddle.float64)
b = a.to(cpu, blocking=not False)
print("#########################case2#########################")
b = a.to("cpu")
print("#########################case3#########################")
b = a.to(device=cpu, dtype=paddle.float64)
print("#########################case4#########################")
b = a.to(paddle.float64)
print("#########################case5#########################")
b = a.to(dtype=paddle.float64)
print("#########################case6#########################")
b = a.to(c)
print("#########################case7#########################")
a = a.to(paddle.float16)
print("#########################case8#########################")
table = a
b = a.to(table.place)
print("#########################case9#########################")
b = a.to(paddle.float32)
print("#########################case10#########################")
device = device2str("cpu")
b = paddle.tensor([-1]).to(paddle.bool)
print("#########################case11#########################")
dtype = paddle.float32
b = a.to(dtype=dtype)
print("#########################case12#########################")
b = a.to(device2str("cpu"))
