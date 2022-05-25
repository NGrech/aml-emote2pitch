import torch
import torch.nn as nn
import torch.nn.functional as F

## U - Net Down
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers=[nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model=nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# U - Net Up
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers= [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
            ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model=nn.Sequential(*layers)

    def forward(self, x, skip_input):
        out=self.model(x)
        out=torch.cat((out, skip_input), 1)
        return out

############################
#       Generator
############################

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()

        self.down1=UNetDown(in_channels, 64, normalize=False)
        self.down2=UNetDown(64, 128)
        self.down3=UNetDown(128, 256)
        self.down4=UNetDown(256, 512, dropout=0.5)
        self.down5=UNetDown(512, 512, dropout=0.5)
        self.down6=UNetDown(512, 512, dropout=0.5)
        self.down7=UNetDown(512, 512, dropout=0.5)
        self.down8=UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1=UNetUp(512, 512, dropout=0.5)
        self.up2=UNetUp(1024, 512, dropout=0.5)
        self.up3=UNetUp(1024, 512, dropout=0.5)
        self.up4=UNetUp(1024, 512, dropout=0.5)
        self.up5=UNetUp(1024, 256)
        self.up6=UNetUp(512, 128)
        self.up7=UNetUp(256, 64)

        self.final=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # the U-Net generator with skip connections from encoder to decoder
        d1=self.down1(x)
        d2=self.down2(d1)
        d3=self.down3(d2)
        d4=self.down4(d3)
        d5=self.down5(d4)
        d6=self.down6(d5)
        d7=self.down7(d6)
        d8=self.down8(d7)

        u1=self.up1(d8, d7)
        u2=self.up2(u1, d6)
        u3=self.up3(u2, d5)
        u4=self.up4(u3, d4)
        u5=self.up5(u4, d3)
        u6=self.up6(u5, d2)
        u7=self.up7(u6, d1)
        out=self.final(u7)

        return out


########################
#    Discriminator
########################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, sigmoid=False):
            ## returns downsampling layers of each discriminator block
            layers= [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))

            if sigmoid:
                layers.append(nn.Sigmoid())
                print('Use Sigmoid')
            return layers 
        
        self.model=nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1,4,padding=1, bias=False)
        )

    def forward(self, img_a, img_b):
        #concatonates images and condition image by channels to produce input
        img_input=torch.cat((img_a, img_b), 1)
        return self.model(img_input)