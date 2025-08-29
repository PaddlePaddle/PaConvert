import numpy as np
import paddle

############################## 相关utils函数，如下 ##############################

def _Tensor_index_copy_(self, dim, index, source):
    if dim == 0:
        return self.scatter_(index, source)

    shape = self.shape

    new_index = []
    for i in range(0, np.prod(shape[:dim])):
        new_index.append(index + i * len(index))
    new_index = paddle.concat(new_index)
    new_self = self.reshape_([-1] + shape[dim+1:])
    new_source = source.reshape([-1] + shape[dim+1:])

    return new_self.scatter_(new_index, new_source).reshape_(shape)

setattr(paddle.Tensor, "index_copy_", _Tensor_index_copy_)
############################## 相关utils函数，如上 ##############################


print("#########################case1#########################")
x = paddle.zeros(5, 3)
t = paddle.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
index = paddle.tensor([0, 4, 2])
x.index_copy_(0, index, t)
print("#########################case2#########################")
x = paddle.zeros(2, 1, 3, 3)
t = paddle.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype="float32",
)
index = paddle.tensor([0, 1, 2])
x.index_copy_(2, index, t)
print("#########################case3#########################")
x = paddle.zeros(5, 3)
t = paddle.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
index = paddle.tensor([0, 4, 2])
y = x.index_copy_(0, index, t)
print("#########################case4#########################")
x = paddle.zeros(2, 1, 3, 3)
t = paddle.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype="float32",
)
index = paddle.tensor([0, 1, 2])
y = x.index_copy_(2, index, t)
print("#########################case5#########################")
x = paddle.zeros(20)
t = paddle.tensor([1, 3, 4, 5], dtype="float32")
index = paddle.tensor([0, 12, 2, 1])
y = x.index_copy_(0, index, t)
