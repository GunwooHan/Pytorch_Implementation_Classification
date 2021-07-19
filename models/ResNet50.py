import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, down_sample=False):
        super(ConvBlock, self).__init__()
        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=2 if down_sample else 1, padding=0 if kernel_size == 1 else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, down_sample=False, first=False):
        super(ResBlock, self).__init__()
        self.down_sample = down_sample
        self.first = first

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, down_sample=down_sample)
        self.conv_block2 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3)
        self.conv_block3 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1)

        if self.down_sample:
            self.shortcut = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, down_sample=True)
        elif self.first:
            self.shortcut = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, input_tensor):
        x = self.conv_block1(input_tensor)
        x = F.relu(x)
        x = self.conv_block2(x)
        x = F.relu(x)
        x = self.conv_block3(x)

        if self.down_sample:
            input_tensor = self.shortcut(input_tensor)
        elif self.first:
            input_tensor = self.shortcut(input_tensor)

        x = x + input_tensor
        x = F.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, color_channel=3, num_classes=10):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=color_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            ResBlock(in_channels=64, out_channels=256, first=True),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256)
        )
        self.conv3_x = nn.Sequential(
            ResBlock(in_channels=256, out_channels=512, down_sample=True),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
        )
        self.conv4_x = nn.Sequential(
            ResBlock(in_channels=512, out_channels=1024, down_sample=True),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
        )
        self.conv5_x = nn.Sequential(
            ResBlock(in_channels=1024, out_channels=2048, down_sample=True),
            ResBlock(in_channels=2048, out_channels=2048),
            ResBlock(in_channels=2048, out_channels=2048)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = ResNet50(color_channel=3, num_classes=10)
    noise = torch.randn(2, 3, 256, 256)
    output = model(noise)
    print(output.size())
