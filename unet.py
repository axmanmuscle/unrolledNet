import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='circular')
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='circular')
        # self.bn2 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.InstanceNorm2d(out_c, affine=True)

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

class decoder_block_630(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        # self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding='same')
        # self.up = nn.Sequential(
        #   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #   nn.Conv2d(in_c, out_c, kernel_size=5, padding='same', padding_mode='circular'),
        # )
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # print(f'decoder block. x shape: {x.shape}, skip shape: {skip.shape}')
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=1, padding='same')
        self.up = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
          nn.Conv2d(in_c, out_c, kernel_size=5, padding='same', padding_mode='circular'),
        )
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # print(f'decoder block. x shape: {x.shape}, skip shape: {skip.shape}')
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet_1ch(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

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

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]
    
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
        outputs = self.crop_to_width(outputs, self.original_width)
        

        return outputs
    
class build_unet_small_1ch(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

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

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]
    
    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
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
        outputs = self.crop_to_width(outputs, self.original_width)        

        return outputs

class build_unet_smaller_1ch(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

        """ Encoder """
        self.e1 = encoder_block(1, 32)
        self.e2 = encoder_block(32, 64)

        """ Bottleneck """
        self.b = conv_block(64, 128)

        """ Decoder """
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]

    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        outputs = self.crop_to_width(outputs, self.original_width)        

        return outputs

class build_unet_smaller(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

        """ Encoder """
        self.e1 = encoder_block(2, 32)
        self.e2 = encoder_block(32, 64)

        """ Bottleneck """
        self.b = conv_block(64, 128)

        """ Decoder """
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 2, kernel_size=1, padding=0)

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]

    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        outputs = self.crop_to_width(outputs, self.original_width)
        
        return outputs
    
class build_unet_small(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

        """ Encoder """
        self.e1 = encoder_block(2, 64)
        self.e2 = encoder_block(64, 128)

        """ Bottleneck """

        self.b = conv_block(128, 256)

        """ Decoder """

        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]
                 
    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        outputs = self.crop_to_width(outputs, self.original_width)
        
        return outputs

class build_unet_smaller_630(nn.Module):
    def __init__(self, width):
        super().__init__()

        """ padding in horizontal direction """

        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)

        hpad = (upperB - width)/2 + 1
        hker = 2*hpad - 1

        self.enlarge = nn.Conv2d(2,2,stride=1,kernel_size=3,padding=(1,int(hpad)))
        self.decimate = nn.Conv2d(2,2,stride=1,kernel_size=(3, int(hker)),padding=(1,0))

        """ Encoder """
        self.e1 = encoder_block(2, 32)
        self.e2 = encoder_block(32, 64)

        """ Bottleneck """
        self.b = conv_block(64, 128)

        """ Decoder """
        self.d3 = decoder_block_630(128, 64)
        self.d4 = decoder_block_630(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 2, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        outputs = self.decimate(outputs)
        
        return outputs

class build_unet(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

        """ Encoder """
        self.e1 = encoder_block(2, 64)
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
        self.outputs = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]

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
        outputs = self.crop_to_width(outputs, self.original_width)
        
        return outputs

class build_unet_smaller_clamp(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.original_width = width

        """ padding in horizontal direction """
        upperB_exp = int( np.ceil( np.log2(width) ) )
        upperB = np.power(2, upperB_exp)
        hpad = (upperB - width)//2
        self.enlarge = nn.ConstantPad2d((hpad, hpad, 0, 0), 0.0)  # (left, right, top, bottom)

        """ Encoder """
        self.e1 = encoder_block(2, 32)
        self.e2 = encoder_block(32, 64)

        """ Bottleneck """
        self.b = conv_block(64, 128)

        """ Decoder """
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 2, kernel_size=1, padding=0)

    def crop_to_width(self, x, width):
        _, _, H, W = x.shape
        start = (W - width) // 2
        return x[:, :, :, start:start + width]

    def forward(self, inputs):

        """ pad to 512 """
        s0 = self.enlarge(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        mag = torch.sqrt(torch.sum(outputs**2, dim=1))
        outputs = outputs / torch.maximum(mag, torch.tensor([1.0]).cuda())

        outputs = self.crop_to_width(outputs, self.original_width)
        
        return outputs

if __name__ == "__main__":
    # f = build_unet(256)
    # summary(f, (1, 256, 256))
    f2 = build_unet_small(256)
    summary(f2, (1, 1, 256, 256))
    # a = torch.rand((1, 1, 256, 256))
    # print(f2(a).shape)

    f3 = build_unet_smaller(84)
    summary(f3, (1,1,380, 84))