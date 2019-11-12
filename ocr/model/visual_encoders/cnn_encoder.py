import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class CNNEncoder(nn.Module):
    """A simple encoder CNN image encoding inspired from Xception net.
    """

    def __init__(self, input_channel, output_channel):
        super(CNNEncoder, self).__init__()
        self.downsample_factor = 8
        self.conv1a = nn.Conv2d(input_channel, 32, kernel_size=(3, 3),
                                stride=(2, 2))
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 3, 2, start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128, 128, 3, 1, start_with_relu=True,
                            grow_first=True)

        self.block3 = Block(128, 256, 3, 2, start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(256, 256, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(256, 256, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(256, 256, 2, 1, start_with_relu=True,
                            grow_first=False)

        self.conv3 = SeparableConv2d(256, output_channel, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(
            self.bn1(self.conv1b(self.conv1a(x))))
        x = self.relu(self.bn2(self.conv2b(self.conv2a(x))))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


if __name__ == '__main__':
    # Call dataloader here to test
    from ocr.data_loader.data_loaders import OCRDataLoader
    from ocr.data_loader.collate import collate_wrapper

    dataloader = OCRDataLoader('../../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print(item[0].size())

    encoder = CNNEncoder(3, 256)
    x = encoder(item[0])
    print(x.size())
