import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np 


class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)
        out = input + out
        return out
    
class Conv_ResBlock(nn.Module):
    def __init__(self, in_dim, conv_dim):
        super(Conv_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlock = ResBlock(conv_dim)

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.ResBlock(out)
        return out
    
########################## Decoder ############################
class DecoderNet(nn.Module):
    def __init__(self, in_dim=64, out_dim=1, num_blocks=10):
        super(DecoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(64) for i in range(num_blocks)])
        self.outputConv = nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        #out = self.deconv1(out)
        #out = F.relu(out)
        #out = self.deconv2(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out


######################### Encoder ##############################
class EncoderNet(nn.Module):
    def __init__(self, in_dim=1, conv_dim = 32, out_dim=32, num_blocks=4):
        super(EncoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(conv_dim) for i in range(num_blocks)])

        self.outputConv = nn.Conv2d(in_channels=conv_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out
    

######################### Lanczos ##############################

''' Pytorch implementation of HomographyNet.
    Reference: https://arxiv.org/pdf/1606.03798.pdf and https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Currently supports translations (2 params)
    The network reads pair of images (tensor x: [B,2*C,W,H])
    and outputs parametric transformations (tensor out: [B,n_params]).'''

def lanczos_kernel(dx, a=3, N=None, dtype=None, device=None):
    '''
    Generates 1D Lanczos kernels for translation and interpolation.
    Args:
        dx : float, tensor (batch_size, 1), the translation in pixels to shift an image.
        a : int, number of lobes in the kernel support.
            If N is None, then the width is the kernel support (length of all lobes),
            S = 2(a + ceil(dx)) + 1.
        N : int, width of the kernel.
            If smaller than S then N is set to S.
    Returns:
        k: tensor (?, ?), lanczos kernel
    '''

    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    D = dx.abs().ceil().int()
    S = 2 * (a + D) + 1  # width of kernel support

    S_max = S.max() if hasattr(S, 'shape') else S

    if (N is None) or (N < S_max):
        N = S

    Z = (N - S) // 2  # width of zeros beyond kernel support

    start = (-(a + D + Z)).min()
    end = (a + D + Z + 1).max()
    x = torch.arange(start, end, dtype=dtype, device=device).view(1, -1) - dx
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / a)

    k = a * sin_px * sin_pxa / px**2  # sinc(x) masked by sinc(x/a)

    return k


def lanczos_shift(img, shift, p=3, a=3):
    '''
    Shifts an image by convolving it with a Lanczos kernel.
    Lanczos interpolation is an approximation to ideal sinc interpolation,
    by windowing a sinc kernel with another sinc function extending up to a
    few nunber of its lobes (typically a=3).

    Args:
        img : tensor (batch_size, channels, height, width), the images to be shifted
        shift : tensor (batch_size, 2) of translation parameters (dy, dx)
        p : int, padding width prior to convolution (default=3)
        a : int, number of lobes in the Lanczos interpolation kernel (default=3)
    Returns:
        I_s: tensor (batch_size, channels, height, width), shifted images
    '''

    dtype = img.dtype

    if len(img.shape) == 2:
        img = img[None, None].repeat(1, shift.shape[0], 1, 1)  # batch of one image
    elif len(img.shape) == 3:  # one image per shift
        assert img.shape[0] == shift.shape[0]
        img = img[None, ]

    # Apply padding

    padder = torch.nn.ReflectionPad2d(p)  # reflect pre-padding
    I_padded = padder(img)

    # Create 1D shifting kernels

    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    k_y = (lanczos_kernel(y_shift, a=a, N=None, dtype=dtype)
           .flip(1)  # flip axis of convolution
           )[:, None, :, None]  # expand dims to get shape (batch, channels, y_kernel, 1)
    k_x = (lanczos_kernel(x_shift, a=a, N=None, dtype=dtype)
           .flip(1)
           )[:, None, None, :]  # shape (batch, channels, 1, x_kernel)

    # Apply kernels

    I_s = torch.conv1d(I_padded,
                       groups=k_y.shape[0],
                       weight=k_y,
                       padding=[k_y.shape[2] // 2, 0])  # same padding
    I_s = torch.conv1d(I_s,
                       groups=k_x.shape[0],
                       weight=k_x,
                       padding=[0, k_x.shape[3] // 2])

    I_s = I_s[..., p:-p, p:-p]  # remove padding

    return I_s.squeeze()  # , k.squeeze()

#########################  ShiftNet ################################
class ShiftNet(nn.Module):
    ''' ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel. '''
    
    def __init__(self, in_channel=1):
        '''
        Args:
            in_channel : int, number of input channels
        '''
        
        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_() # init the weights with the identity transformation

    def forward(self, x):
        '''
        Registers pairs of images with sub-pixel shifts.
        Args:
            x : tensor (B, 2*C_in, H, W), input pairs of images
        Returns:
            out: tensor (B, 2), translation params
        '''

        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(1, 2)).view(-1, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(1, 2)).view(-1, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    def transform(self, theta, I, device="cpu"):
        '''
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        '''

        self.theta = theta
        new_I = lanczos_shift(img=I.transpose(0, 1),
                              shift=self.theta.flip(-1),  # (dx, dy) from register_batch -> flip
                              a=3, p=5)[:, None]
        return new_I
    
    
###########################################################################
class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        #nn.init.kaiming_normal_(self.conv1.weight)
        #nn.init.kaiming_normal_(self.conv2.weight)
        
        
    def forward(self, input):
        out = self.conv1(input)
        #print('conv1: {}'.format(out.size()))
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out
    
class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mode):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if mode == "maxpool":
            self.final = lambda x: F.max_pool2d(x, kernel_size=2)
        elif mode == "bilinear":
            self.final = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            raise Exception('mode must be maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class FNet(nn.Module):
    def __init__(self, in_dim=2):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, mode="maxpool")
        self.convPool2 = FNetBlock(32, 64, mode="maxpool")
        self.convPool3 = FNetBlock(64, 128, mode="maxpool")
        self.convBinl1 = FNetBlock(128, 256, mode="bilinear")
        self.convBinl2 = FNetBlock(256, 128, mode="bilinear")
        self.convBinl3 = FNetBlock(128, 64, mode="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        
        #nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        self.out = torch.tanh(out)*5.
        #self.out.retain_grad()
        return self.out