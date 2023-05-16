import paddle


class LeNet(paddle.nn.Layer):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6,
            kernel_size=5, stride=1, padding=2, bias_attr=False)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16,
            kernel_size=5, stride=1, bias_attr=False)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5,
            out_features=120, bias_attr=False)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84,
            bias_attr=False)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10,
            bias_attr=False)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.max_pool2(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.linear2(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.linear3(x)
        return x

    def _init_weight(self):
        conv1_weights = paddle.ones(shape=(6, 1, 5, 5))
        paddle.assign(conv1_weights, output=self.conv1.weight)
        conv2_weights = paddle.ones(shape=(16, 6, 5, 5))
        paddle.assign(conv2_weights, output=self.conv2.weight.data)
        linear1_weights = paddle.ones(shape=(120, 400))
        paddle.assign(linear1_weights, output=self.linear1.weight.data)
        linear2_weights = paddle.ones(shape=(84, 120))
        paddle.assign(linear2_weights, output=self.linear2.weight.data)
        linear3_weights = paddle.ones(shape=(10, 84))
        paddle.assign(linear3_weights, output=self.linear3.weight.data)


def train(model):
    loss_vec = []
    model.train()
    epochs = 1
    for epoch in range(epochs):
        x_data = paddle.ones(shape=[1, 1, 28, 28])
        y_data = paddle.to_tensor(data=[5])
        predicts = model(x_data)
        print('torch_file')
        print('x_data: ', x_data[0][0][0])
        print('y_data: ', y_data)
        print('predicts: ', predicts)
        loss = paddle.nn.functional.cross_entropy(input=predicts, label=y_data)
        print('loss: ', loss)
        loss_vec.append(loss)
    return loss_vec


def lenet_loss_result():
    model = LeNet()
    loss_vec = train(model)
    return loss_vec


if __name__ == '__main__':
    print(lenet_loss_result())
