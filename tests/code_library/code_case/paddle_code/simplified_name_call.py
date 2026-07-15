import paddle

paddle.enable_compat(level=2)
a = paddle.tensor([1])
b = paddle.tensor([2])
print("#########################case1#########################")
func = paddle.add
paddle.add(a, b)
print("#########################case2#########################")
func = paddle.matmul
paddle.matmul(a, b)
