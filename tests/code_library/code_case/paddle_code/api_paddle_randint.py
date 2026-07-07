import paddle

print("#########################case1#########################")
a = paddle.randint(2, 5, [3, 4], device=paddle.device("cuda"))
print("#########################case2#########################")
paddle.randint(10, [2, 2])
print("#########################case3#########################")
a, b = 2, 25
a = paddle.randint(a, b, [3, 4], device=paddle.device("cuda"))
