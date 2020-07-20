import torch
import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):
    """6-scale UNet++."""
    def __init__(self):
        super().__init__()
        self.InputConv = InputConv(32)
        self.ECA = nn.ModuleList([ECA(32) for _ in range(54)])
        self.DownConv = nn.ModuleList([DownConv(32, 32) for _ in range(5)])
        self.UpConv_64 = nn.ModuleList([UpConv(64, 32) for _ in range(5)])
        self.UpConv_96 = nn.ModuleList([UpConv(96, 32) for _ in range(4)])
        self.UpConv_128 = nn.ModuleList([UpConv(128, 32) for _ in range(3)])
        self.UpConv_160 = nn.ModuleList([UpConv(160, 32) for _ in range(2)])
        self.UpConv_192 = UpConv(192, 32)
        self.OutConv = nn.ModuleList([OutConv(32) for _ in range(5)])
        
    def forward(self, x, order=5, mod='val'):
        x00 = self.InputConv(x)
        
        x10 = self.DownConv[0](self.ECA[0](x00))
        x01 = self.UpConv_64[0](self.ECA[1](x00),
                                self.ECA[2](x10))
        
        if (mod == 'val') and (order == 1):
            out1 = self.OutConv[0](self.ECA[3](x01))
            return out1 + x
        
        x20 = self.DownConv[1](self.ECA[4](x10))
        x11 = self.UpConv_64[1](self.ECA[5](x10),
                                self.ECA[6](x20))
        x02 = self.UpConv_96[0](torch.cat((self.ECA[7](x00), 
                                           self.ECA[8](x01)), dim=1),
                                self.ECA[9](x11))
        
        if (mod == 'val') and (order == 2):
            out2 = self.OutConv[1](self.ECA[10](x02))
            return out2 + x
        
        x30 = self.DownConv[2](self.ECA[11](x20))
        x21 = self.UpConv_64[2](self.ECA[12](x20), 
                                self.ECA[13](x30))
        x12 = self.UpConv_96[1](torch.cat((self.ECA[14](x10),
                                           self.ECA[15](x11)), dim=1),
                                self.ECA[16](x21))
        x03 = self.UpConv_128[0](torch.cat((self.ECA[17](x00),
                                            self.ECA[18](x01),
                                            self.ECA[19](x02)), dim=1),
                                self.ECA[20](x12))

        if (mod == 'val') and (order == 3):
            out3 = self.OutConv[2](self.ECA[21](x03))
            return out3 + x
        
        x40 = self.DownConv[3](self.ECA[22](x30))
        x31 = self.UpConv_64[3](self.ECA[23](x30), 
                                self.ECA[24](x40))
        x22 = self.UpConv_96[2](torch.cat((self.ECA[25](x20),
                                           self.ECA[26](x21)), dim=1),
                                self.ECA[27](x31))
        x13 = self.UpConv_128[1](torch.cat((self.ECA[28](x10),
                                            self.ECA[29](x11),
                                            self.ECA[30](x12)), dim=1),
                                 self.ECA[31](x22))
        x04 = self.UpConv_160[0](torch.cat((self.ECA[32](x00),
                                            self.ECA[33](x01),
                                            self.ECA[34](x02),
                                            self.ECA[35](x03)), dim=1),
                                self.ECA[36](x13))
              
        if (mod == 'val') and (order == 4):
            out4 = self.OutConv[3](self.ECA[37](x04))
            return out4 + x
        
        x50 = self.DownConv[4](self.ECA[38](x40))
        x41 = self.UpConv_64[4](self.ECA[39](x40),
                                x50)
        x32 = self.UpConv_96[3](torch.cat((self.ECA[40](x30),
                                           self.ECA[41](x31)), dim=1),
                                x41)
        x23 = self.UpConv_128[2](torch.cat((self.ECA[42](x20),
                                            self.ECA[43](x21),
                                            self.ECA[44](x22)), dim=1),
                                 x32)
        x14 = self.UpConv_160[1](torch.cat((self.ECA[45](x10),
                                            self.ECA[46](x11),
                                            self.ECA[47](x12),
                                            self.ECA[48](x13)), dim=1),
                                 x23)
        x05 = self.UpConv_192(torch.cat((self.ECA[49](x00),
                                         self.ECA[50](x01),
                                         self.ECA[51](x02),
                                         self.ECA[52](x03),
                                         self.ECA[53](x04)), dim=1),
                              x14)
        
        if (mod == 'val') and (order == 5):
            out5 = self.OutConv[4](x05)
            return out5 + x
        
        if mod == 'tra':
            out1 = self.OutConv[0](self.ECA[3](x01))
            out2 = self.OutConv[1](self.ECA[10](x02))
            out3 = self.OutConv[2](self.ECA[21](x03))
            out4 = self.OutConv[3](self.ECA[37](x04))
            out5 = self.OutConv[4](x05)
            return out1 + x, out2 + x, out3 + x, out4 + x, out5 + x


class InputConv(nn.Module):
    def __init__(self, outch):
        super().__init__()
        self.InputConv = nn.Sequential(
            nn.Conv2d(1, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.InputConv(x)


class DownConv(nn.Module):
    def __init__(self, inch, outch):
        super().__init__()
        self.DownConv = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outch, outch, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.DownConv(x)


class UpConv(nn.Module):
    def __init__(self, inch, outch):
        super().__init__()
        self.UpConv = nn.Sequential(
            nn.ConvTranspose2d(outch, outch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
            )
        self.DoubleSeparableConv = nn.Sequential(SeparableConv2d(inch, outch),
                                                 nn.ReLU(inplace=True),
                                                 SeparableConv2d(outch, outch),
                                                 nn.ReLU(inplace=True)
                                                 )

    def forward(self, x, x_l):
        x_l = self.UpConv(x_l)
        return self.DoubleSeparableConv(torch.cat((x, x_l), dim=1))
   

class SeparableConv2d(nn.Module):
    def __init__(self, inch, outch):
        super().__init__()
        self.SeparableConv = nn.Sequential(nn.Conv2d(inch, inch, kernel_size=3, padding=1, groups=inch), # groups=inch: each inch is convolved with its own filter
                                           nn.Conv2d(inch, outch, kernel_size=1, groups=1) # then point-wise
                                           )
    
    def forward(self, x):
        return self.SeparableConv(x)


class ECA(nn.Module):
    def __init__(self, inch):
        super().__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1) # B inch 1 1
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.ave_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class OutConv(nn.Module):
    def __init__(self, inch):
        super().__init__()
        self.OutConv = SeparableConv2d(inch, 1)

    def forward(self, x):
        return self.OutConv(x)
