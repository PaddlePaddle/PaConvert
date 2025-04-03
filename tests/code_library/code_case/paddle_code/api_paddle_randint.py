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
a = paddle.randint(low=2, high=5, shape=[3, 4])
print("#########################case2#########################")
paddle.randint(low=0, high=10, shape=[2, 2])
print("#########################case3#########################")
a, b = 2, 25
a = paddle.randint(low=a, high=b, shape=[3, 4])
