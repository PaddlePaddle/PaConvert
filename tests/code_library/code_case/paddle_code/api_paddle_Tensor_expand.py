import paddle

print("#########################case1#########################")
x = paddle.to_tensor(data=[[1], [2], [3]])
y = x.expand(shape=[3, 4])
print("#########################case2#########################")
x = paddle.to_tensor(data=[[1], [2], [3]])
y = x.expand(shape=(3, 4))
