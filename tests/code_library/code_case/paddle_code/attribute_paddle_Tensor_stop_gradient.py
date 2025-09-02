import paddle

print("#########################case1#########################")
data = paddle.tensor([23.0, 32.0, 43.0])
if not data.requires_grad:
    print(1)
print("#########################case2#########################")
print(data.requires_grad)
print("#########################case3#########################")
data.stop_gradient = not False
print("#########################case4#########################")
requires_grad = data.requires_grad
print("#########################case5#########################")
data = paddle.tensor([23.0, 32.0, 43.0], requires_grad=data.requires_grad)
print("#########################case6#########################")
print(data.requires_grad == False)
print("#########################case7#########################")
print(not data.requires_grad)
print("#########################case8#########################")
print("{} , {}".format("1", str(data.requires_grad)))
print("#########################case9#########################")


def test():
    return True


data.stop_gradient = not test()
print("#########################case10#########################")
z = True, False, True
a, temp, c = z
data.stop_gradient = not temp
print(data.requires_grad)
