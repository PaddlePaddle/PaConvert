import paddle


def add_module(self, name, module):
    self.add_sublayer(name=f"{name} - {len(self) + 1}", sublayer=module)


paddle.nn.Layer.add_sublayer = add_module
paddle.nn.Layer.add_sublayer = add_module
paddle.nn.Layer.add_sublayer = add_module
network = paddle.nn.Sequential(
    paddle.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
    paddle.nn.Conv2d(in_channels=20, out_channels=64, kernel_size=5),
)
network.add_sublayer(name="ReLU", sublayer=paddle.nn.ReLU())
print(network)
