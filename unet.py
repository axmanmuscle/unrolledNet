import torch
import torch.nn as nn
from torchinfo import summary

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self, width):
        super().__init__()

        """ padding in horizontal direction """
        hpad = (512 - width)/2 + 1
        hker = 2*hpad - 1
        self.enlarge = nn.Conv2d(1,1,stride=1,kernel_size=3,padding=(1,int(hpad)))
        self.decimate = nn.Conv2d(1,1,stride=1,kernel_size=(3, int(hker)),padding=(1,0))

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        #self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        #self.b = conv_block(512, 1024)
        self.b = conv_block(256, 512)

        """ Decoder """
        #self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        #s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        #d1 = self.d1(b, s4)
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        outputs = self.decimate(outputs)
        

        return outputs
    
class build_unet_small(nn.Module):
    def __init__(self, width):
        super().__init__()

        """ padding in horizontal direction """
        hpad = (512 - width)/2 + 1
        hker = 2*hpad - 1
        self.enlarge = nn.Conv2d(1,1,stride=1,kernel_size=3,padding=(1,int(hpad)))
        self.decimate = nn.Conv2d(1,1,stride=1,kernel_size=(3, int(hker)),padding=(1,0))

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        # self.e3 = encoder_block(128, 256)
        #self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        #self.b = conv_block(512, 1024)
        #self.b = conv_block(256, 512)
        # self.b = conv_block(128, 256)
        self.b = conv_block(128, 256)

        """ Decoder """
        #self.d1 = decoder_block(1024, 512)
        # self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ pad to 512 """
        # s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        # s3, p3 = self.e3(p2)
        #s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        #d1 = self.d1(b, s4)
        # d2 = self.d2(b, s3)
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        # outputs = self.decimate(outputs)
        

        return outputs

if __name__ == "__main__":
    # f = build_unet(256)
    # summary(f, (1, 256, 256))
    f2 = build_unet_small(256)
    summary(f2, (1, 1, 256, 256))
    a = torch.rand((1, 1, 256, 256))
    print(f2(a).shape)
