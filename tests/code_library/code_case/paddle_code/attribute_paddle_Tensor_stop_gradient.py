import paddle

print("#########################case1#########################")
data = paddle.tensor([23.0, 32.0, 43.0])
if not not data.stop_gradient:
    print(1)
print("#########################case2#########################")
print(not data.stop_gradient)
print("#########################case3#########################")
data.stop_gradient = not False
print("#########################case4#########################")
requires_grad = not data.stop_gradient
print("#########################case5#########################")
data = paddle.tensor(
    [23.0, 32.0, 43.0], requires_grad=not data.stop_gradient
)
print("#########################case6#########################")
print((not data.stop_gradient) == False)
print("#########################case7#########################")
print(not not data.stop_gradient)
print("#########################case8#########################")
print("{} , {}".format("1", str(not data.stop_gradient)))
print("#########################case9#########################")


def test():
    return True


data.stop_gradient = not test()
print("#########################case10#########################")
z = True, False, True
a, temp, c = z
data.stop_gradient = not temp
print(not data.stop_gradient)
