import torch.nn as nn



class NET(nn.Module):
    def __init__(self, latent_size):
        super(NET, self).__init__()
        # 定义隐藏层
        self.hidden1 = nn.Linear(latent_size, 10).double()
        self.active1 = nn.ReLU().double()

        self.hidden2 = nn.Linear(10, 10).double()
        self.active2 = nn.ReLU().double()

        self.hidden3 = nn.Linear(10, 20).double()
        self.active3 = nn.ReLU().double()

        self.hidden4 = nn.Linear(20, 10).double()
        self.active4 = nn.ReLU().double()

        # 定义预测回归层
        self.regression = nn.Linear(10, 1).double()

    # 定义网络的向前传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        x = self.hidden3(x)
        x = self.active3(x)
        x = self.hidden4(x)
        x = self.active4(x)
        output = self.regression(x)
        return output
