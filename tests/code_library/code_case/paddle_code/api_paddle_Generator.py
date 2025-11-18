import paddle

print("#########################case1#########################")
g_cpu = paddle.Generator()
print("#########################case2#########################")
g_cpu = paddle.Generator(device="cpu")
print("#########################case3#########################")
g_cpu = paddle.Generator("cpu")
print("#########################case4#########################")
g_cuda = paddle.Generator("cuda")
print("#########################case5#########################")
g_cuda = paddle.Generator(device="cuda")
