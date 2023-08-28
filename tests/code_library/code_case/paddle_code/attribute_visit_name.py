import paddle


class A(paddle.nn.Layer):

    def __init__(self, data: paddle.Tensor):
        pass


def func1(data: paddle.Tensor):
    data(x)
    data['id']
    func(data)


def func2() ->paddle.Tensor:
    pass


def func3(dtype='float32'):
    pass


isinstance(x, paddle.Tensor)
setattr(paddle.Tensor, 'add', add_func)
