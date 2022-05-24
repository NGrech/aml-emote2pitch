## Modify this 

def make_conv_block(_in, out, batch_norm=True):
    kernel = (4,4)
    stride=2
    pad = 1
    layers = []
    layers.append(nn.Conv2d(_in, out, kernel_size=kernel, stride=stride, padding=pad))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out))
    layers.append(nn.LeakyReLU(0.2,inplace=True))
    return nn.Sequential(*layers)
 
def make_decov_block(_in, out, batch_norm=True):
    kernel = (4,4)
    stride=2
    pad = 1
    layers = []
    layers.append(nn.ConvTranspose2d(_in, out, kernel_size=kernel, stride=stride, padding=pad))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out))
    layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)