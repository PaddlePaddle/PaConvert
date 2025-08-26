import paddle

a = paddle.tensor([1])
b = paddle.tensor([2])
print("#########################case1#########################")
func = paddle.add
paddle.add(x=a, y=paddle.to_tensor(b))
print("#########################case2#########################")
func = paddle.matmul
paddle.matmul(x=a, y=b)
