import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.d1 = double_conv(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = double_conv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = double_conv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.bottleneck = double_conv(256, 512)

        self.att3 = AttentionBlock(256, 256, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        self.att1 = AttentionBlock(64, 64, 32)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = double_conv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = double_conv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        bn = self.bottleneck(self.p3(d3))

        # u3 = self.u3(bn)
        # u3 = torch.cat([u3, d3], dim=1)
        # u3 = self.up3(u3)
        u3 = self.u3(bn)
        d3 = self.att3(u3, d3)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3)

        # u2 = self.u2(u3)
        # u2 = torch.cat([u2, d2], dim=1)
        # u2 = self.up2(u2)
        u2 = self.u2(u3)
        d2 = self.att2(u2, d2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)

        # u1 = self.u1(u2)
        # u1 = torch.cat([u1, d1], dim=1)
        # u1 = self.up1(u1)
        u1 = self.u1(u2)
        d1 = self.att1(u1, d1)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1)

        return self.out(u1)