from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.a = channel // reduction // 3
        self.b = channel // reduction - self.a*2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, self.a),
                nn.BatchNorm2d(self.a),
                nn.ReLU(inplace=True),
                nn.Linear(self.a, channel),
        )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(2)
	self.fc1 = nn.Sequential(
                nn.Conv2d(channel, self.a, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.a),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.a, channel, kernel_size=2, bias=True),
        )
        self.avg_pool2 = nn.AdaptiveAvgPool2d(4)
	self.fc2 = nn.Sequential(
                nn.Conv2d(channel, self.b, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.b),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.b, channel, kernel_size=4, bias=True),
        )
	self.sig = nn.Sequential(
		nn.BatchNorm2d(channel),
       		nn.Sigmoid()
	)

    def forward(self, x):
        d, c, _, _ = x.size()
        y = self.avg_pool(x).view(d, c)
        y = self.fc(y).view(d, c, 1, 1)
	y1 = self.avg_pool1(x)
        y1 = self.fc1(y1).view(d, c, 1, 1)
        y2 = self.avg_pool2(x)
        y2 = self.fc2(y2).view(d, c, 1, 1)
	y = y +y1+y2
        y = self.sig(y)
        return x * y
