import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.pool1_2=nn.MaxPool2d(kernel_size=2)
        self.pool1_3=nn.AvgPool2d(kernel_size=4)
        self.pool1_4=nn.AvgPool2d(kernel_size=8)
        self.pool1_5=nn.AvgPool2d(kernel_size=16)
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.pool2_3=nn.MaxPool2d(kernel_size=2)
        self.pool2_4=nn.AvgPool2d(kernel_size=4)
        self.pool2_5=nn.AvgPool2d(kernel_size=8)
        self.conv3=nn.Sequential(
            nn.Conv2d(48,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.pool3_4=nn.MaxPool2d(kernel_size=2)
        self.pool3_5=nn.AvgPool2d(kernel_size=4)
        self.conv4=nn.Sequential(
            nn.Conv2d(112,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.pool4_5=nn.MaxPool2d(kernel_size=2)
        self.conv5a=nn.Sequential(
            nn.Conv2d(240,20,3,1,1),
            nn.BatchNorm2d(20),
#             nn.Tanh(),
        )
        self.conv5b=nn.Sequential(
            nn.Conv2d(240,20,3,1,1),
            nn.BatchNorm2d(20),
#             nn.Tanh(),
        )
        self.gp=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(20,1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self,x):
        # print(x.type())
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1_2(x1))
        x3 = self.conv3(torch.cat((self.pool1_3(x1),self.pool2_3(x2)),dim=1))
        x4 = self.conv4(torch.cat((self.pool1_4(x1),self.pool2_4(x2),self.pool3_4(x3)),dim=1))
        mu = self.conv5a(torch.cat((self.pool1_5(x1),self.pool2_5(x2),self.pool3_5(x3),self.pool4_5(x4)),dim=1))
        logvar = self.conv5b(torch.cat((self.pool1_5(x1),self.pool2_5(x2),self.pool3_5(x3),self.pool4_5(x4)),dim=1))
        x = self.gp(self.reparameterize(mu, logvar))
        output = x.view(len(x),-1)
        output = self.fc(output)
        return output,mu,logvar
