# Lab 8: Image Classification

:man_student: Charles

## 实验概述

通过完成 `CIFAR-10` 上的图像分类任务，学习一些经典的计算机视觉网络结构及其变体，以及一些优化训练方式。

## 实验过程及结果

### Baseline

代码已提供的Baseline表现如下：

<img src="./images/baseline_1" alt="baseline1" width="500;" /> 

### 自定义网络1

参考 [vgg论文](https://arxiv.org/abs/1409.1556) 设计。考虑到本实验用的 `CIFAR-10` 数据集大小和其中图片分辨率都远小于论文使用的 `ImageNet` ，故选择参考相对简单的 `vgg11` 结构，同时将卷积层的输出通道数和全连接层的神经元数减小（取一半或1/4）：

```python
class MyNet1(nn.Module):
    """
    参考vgg11
    """
    
    def __init__(self):
        super(MyNet1, self).__init__()

        # 卷积模块1
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积模块2
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 卷积模块3
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 卷积模块4
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 卷积模块5
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(256 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 10)

        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```

训练结果：

<img src="./images/mynet1_1" alt="mynet1_1" width="500;" /> 

> 相关超参：
>
> ```python
> batch_size = 64
> optimizer = optim.Adam(net.parameters(), lr=0.001)
> n_epoch = 10
> ```

## 自定义网络2

参考 [ResNet论文](https://arxiv.org/abs/1512.03385) 设计。同样地，由于数据集较小，选择参考相对简单的 `ResNet18` 设计

```python
class MyNet2(nn.Module):
    """
    参考ResNet18
    """

    def __init__(self):
        super(MyNet2, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差块层
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, input_size, output_size, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock(input_size, output_size, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(output_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, input_size, output_size, stride=1):
        super(ResBlock, self).__init__()

        # 主路径的卷积层
        self.conv1 = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_size)
        self.conv2 = nn.Conv2d(
            output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(output_size)

        # 残差路径
        self.shortcut = nn.Sequential()
        if stride != 1 or input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_size, output_size, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(output_size),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out
```

训练结果：

<img src="./images/mynet2_1" alt="mynet2" width="500;" /> 

> 相关超参：
>
> ```python
> batch_size = 128
> optimizer = optim.Adam(net.parameters(), lr=0.001)
> n_epoch = 10
> ```
