import paddle


def add_module(self, name, module):
    self.add_module(f"{name} - {len(self) + 1}", module)


paddle.nn.Module.add_module = add_module
paddle.nn.Module.add_module = add_module
paddle.nn.Module.add_module = add_module
network = paddle.nn.Sequential(paddle.nn.Conv2d(1, 20, 5), paddle.nn.Conv2d(20, 64, 5))
network.add_module("ReLU", paddle.nn.ReLU())
print(network)
