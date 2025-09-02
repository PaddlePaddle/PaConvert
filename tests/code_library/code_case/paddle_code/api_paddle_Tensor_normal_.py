import paddle

print("#########################case1#########################")
a = paddle.tensor([1, 3, 4, 9, 0.5, 1.5])
a = a.normal_(0.2, 0.3)
