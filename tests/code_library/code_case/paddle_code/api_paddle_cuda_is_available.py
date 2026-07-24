import paddle

paddle.enable_compat(level=2)
print("#########################case1#########################")
paddle.cuda.is_available()
