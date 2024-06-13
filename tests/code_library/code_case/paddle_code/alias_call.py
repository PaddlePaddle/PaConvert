import paddle
a = paddle.to_tensor(data=[1])
b = paddle.to_tensor(data=[2])
print('#########################case1#########################')
func = paddle.add
paddle.add(x=a, y=paddle.to_tensor(b))
print('#########################case2#########################')
func = paddle.matmul
paddle.matmul(x=a, y=b)
