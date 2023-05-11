import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120, bias=False)
        self.linear2 = torch.nn.Linear(in_features=120, out_features=84, bias=False)
        self.linear3 = torch.nn.Linear(in_features=84, out_features=10, bias=False)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def _init_weight(self):
        conv1_weights = torch.ones((6, 1, 5, 5))
        self.conv1.weight.copy_(conv1_weights)
        conv2_weights = torch.ones((16, 6, 5, 5))
        self.conv2.weight.data.copy_(conv2_weights)
        linear1_weights = torch.ones((120, 400))
        self.linear1.weight.data.copy_(linear1_weights)
        linear2_weights = torch.ones((84, 120))
        self.linear2.weight.data.copy_(linear2_weights)
        linear3_weights = torch.ones((10, 84))
        self.linear3.weight.data.copy_(linear3_weights)


def train(model):
    loss_vec = []
    model.train()
    epochs = 1
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        x_data = torch.ones([1,1,28,28])
        y_data = torch.tensor([5])
        predicts = model(x_data)
        print("torch_file")
        print("x_data: ", x_data[0][0][0])
        print("y_data: ", y_data)
        print("predicts: ", predicts)
        loss = F.cross_entropy(predicts, y_data)
        print("loss: ", loss)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        loss_vec.append(loss)
    return loss_vec

def lenet_loss_result():
    model = LeNet()
    loss_vec = train(model)
    return loss_vec

if __name__=="__main__":
    print(lenet_loss_result())
