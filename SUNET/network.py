import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(), nn.Conv2d(channel, channel, 3, padding=1),
                                    nn.BatchNorm2d(channel), nn.ReLU(),
                                    nn.Conv2d(channel, channel, 3, padding=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.layer1(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_dim=23):
        super(Unet, self).__init__()
        self.layer_down1 = nn.Sequential(nn.Conv2d(input_dim, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.layer_down2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer_down3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer_down4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.upscale1 = nn.PixelShuffle(2)
        self.layer_up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        # self.att1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        #                           nn.Conv2d(256, 256, 3, padding=1))
        self.upscale2 = nn.PixelShuffle(2)
        self.layer_up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        # self.att2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        #                           nn.Conv2d(128, 128, 3, padding=1))
        self.upscale3 = nn.PixelShuffle(2)
        self.layer_up3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                       nn.Conv2d(16, 1, 3, padding=1))
        # self.att3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        #                           nn.Conv2d(64, 1, 3, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, cloth, seg):
        res = 1 - seg[:, 0:1, :, :]
        # cloth = res * cloth
        res = res - 0.5
        x = torch.cat((cloth, seg), 1)
        x1 = self.layer_down1(x)
        x2 = self.pool(x1)
        x2 = self.layer_down2(x2)
        x3 = self.pool(x2)
        x3 = self.layer_down3(x3)
        x4 = self.pool(x3)
        x4 = self.layer_down4(x4)
        x = self.upscale1(x4)
        x = torch.cat((x, x3), 1)
        x = self.layer_up1(x)
        x = self.upscale2(x)
        x = torch.cat((x, x2), 1)
        x = self.layer_up2(x)
        x = self.upscale3(x)
        x = torch.cat((x, x1), 1)
        x = self.layer_up3(x)
        return self.sig(x + res)


class Unet_heat(nn.Module):
    def __init__(self):
        super(Unet_heat, self).__init__()
        self.layer_down1 = nn.Sequential(nn.Conv2d(28, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.layer_down2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer_down3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer_down4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.upscale1 = nn.PixelShuffle(2)
        self.layer_up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.att1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1))
        self.upscale2 = nn.PixelShuffle(2)
        self.layer_up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.att2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding=1))
        self.upscale3 = nn.PixelShuffle(2)
        self.layer_up3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                       nn.Conv2d(16, 1, 3, padding=1))
        self.att3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, cloth, heatmap):
        x = torch.cat((cloth, heatmap), 1)
        x1 = self.layer_down1(x)
        x2 = self.pool(x1)
        x2 = self.layer_down2(x2)
        x3 = self.pool(x2)
        x3 = self.layer_down3(x3)
        x4 = self.pool(x3)
        x4 = self.layer_down4(x4)
        x = self.upscale1(x4)
        x = torch.cat((x, x3), 1)
        x = self.layer_up1(x) * (1 + self.att1(x))
        x = self.upscale2(x)
        x = torch.cat((x, x2), 1)
        x = self.layer_up2(x) * (1 + self.att2(x))
        x = self.upscale3(x)
        x = torch.cat((x, x1), 1)
        x = self.layer_up3(x) * (1 + self.att3(x))
        return x


class Unet_2(nn.Module):
    def __init__(self, input_dim=23):
        super(Unet_2, self).__init__()
        self.layer_down1 = nn.Sequential(nn.Conv2d(input_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.layer_down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer_down3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.layer_down4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                         nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer_down5 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
                                         nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU())
        self.upscale1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.layer_up1 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        # self.att1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        #                           nn.Conv2d(256, 256, 3, padding=1))
        self.upscale2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.layer_up2 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        # self.att2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        #                           nn.Conv2d(128, 128, 3, padding=1))
        self.upscale3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.layer_up3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.upscale4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.layer_up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 1, 1))

        # self.att3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        #                           nn.Conv2d(64, 1, 3, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, cloth, seg):
        # res = 1 - seg[:, 0:1, :, :]
        # cloth = res * cloth
        # res = res - 0.5
        x = torch.cat((cloth, seg), 1)
        x1 = self.layer_down1(x)
        x2 = self.pool(x1)
        x2 = self.layer_down2(x2)
        x3 = self.pool(x2)
        x3 = self.layer_down3(x3)
        x4 = self.pool(x3)
        x4 = self.layer_down4(x4)
        x5 = self.pool(x4)
        x5 = self.layer_down5(x5)
        x = self.upscale1(x5)
        x = torch.cat((x, x4), 1)
        x = self.layer_up1(x)
        x = self.upscale2(x)
        x = torch.cat((x, x3), 1)
        x = self.layer_up2(x)
        x = self.upscale3(x)
        x = torch.cat((x, x2), 1)
        x = self.layer_up3(x)
        x = self.upscale4(x)
        x = torch.cat((x, x1), 1)
        x = self.layer_up4(x)

        return self.sig(x)
