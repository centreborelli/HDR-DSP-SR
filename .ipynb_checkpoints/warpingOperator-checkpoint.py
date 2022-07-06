import random
import torch 
import torch.nn as nn

import numpy as np 
import scipy
from scipy.ndimage.morphology import binary_dilation
from torch.autograd import Variable
import iio

def base_detail_decomp(samples, gaussian_filter):
    #samplesLR: b, num_im, h, w
    b, num_im, h, w = samples.shape
    base   = gaussian_filter(samples.view(-1,1,h,w)).view(*samples.shape)
    detail = samples - base
    return base, detail #b, num_im, h, w


class BlurLayer(nn.Module):
    def __init__(self):
        super(BlurLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(size//2), 
            nn.Conv2d(1, 1, size, stride=1, padding=0, bias=None, groups=1)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        k = iio.read('blur_kernel.tiff').squeeze()
        size = k.shape[0]
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.required_grad = False

class GaussianLayer(nn.Module):
    def __init__(self, sigma=1):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(5), 
            nn.Conv2d(1, 1, 11, stride=1, padding=0, bias=None, groups=1)
        )

        self.sigma = sigma
        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((11,11))
        n[5,5] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.required_grad = False

Gaussian_Filter = GaussianLayer(sigma=1).to('cuda')

class TVL1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVL1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
    
        count_h = self._tensor_size(x[...,1:,:])
        count_w = self._tensor_size(x[...,:,1:])        
    
        h_tv = torch.abs((x[...,1:,:]-x[...,:h_x-1,:])).sum()
        w_tv = torch.abs((x[...,:,1:]-x[...,:,:w_x-1])).sum()
        #print("h,w:", h_tv, w_tv)
        return self.TVLoss_weight*(h_tv/count_h+w_tv/count_w)/batch_size
        #return self.TVLoss_weight*(h_tv+w_tv)/batch_size
        
    def _tensor_size(self,t):
        return t.size()[-3]*t.size()[-2]*t.size()[-1]


class WarpedLoss(nn.Module):
    def __init__(self, p = 1, interpolation = 'bilinear'):
        super(WarpedLoss, self).__init__()
        if p == 1:
            self.criterion = nn.L1Loss(reduction='mean') #change to reduction = 'mean'
        if p == 2:
            self.criterion = nn.MSELoss(reduction='mean')
        self.interpolation = interpolation
    def cubic_interpolation(self, A, B, C, D, x):
        a,b,c,d = A.size()
        x = x.view(a,1,c,d)#.repeat(1,3,1,1)
        return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        if torch.sum(flo*flo) == 0:
            return x
        else:
            
            B, C, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = grid.cuda()
            #print(grid.shape)
            vgrid = Variable(grid) + flo.cuda()

            if self.interpolation == 'bilinear':
                # scale grid to [-1,1] 
                vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
                vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

                vgrid = vgrid.permute(0,2,3,1)        
                output = nn.functional.grid_sample(x, vgrid,align_corners = True)

            if self.interpolation == 'bicubicTorch':
                # scale grid to [-1,1] 
                vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
                vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

                vgrid = vgrid.permute(0,2,3,1)        
                output = nn.functional.grid_sample(x, vgrid,align_corners = True,mode = 'bicubic')
                
                #mask = torch.ones(x.size()).cuda()
                #mask = nn.functional.grid_sample(mask, vgrid,align_corners = True,mode = 'bicubic')

                #mask[mask < 0.9999] = 0
                #mask[mask > 0] = 1
            return output#, mask

    def forward(self, input, target, flow, losstype = 'L1', masks = None):
        # Warp input on target
        warped = self.warp(target, flow)

        input_ = input[...,5:-5,5:-5]
        warped_ = warped[...,5:-5,5:-5]
        if losstype == 'HighRes-net':
            warped_ = warped_/torch.sum(warped_, dim = (2,3), keepdims = True)*torch.sum(input_, dim = (2,3), keepdims = True)
        if losstype == 'Detail':
            _, warped_ =  base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ =  base_detail_decomp(input_, Gaussian_Filter)
            
        if losstype == 'DetailReal':
            _, warped_ =  base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ =  base_detail_decomp(input_, Gaussian_Filter)
            
            masks = masks[...,2:-2,2:-2]
            
            warped_ = warped_ * masks[:,:1] * masks[:,1:]
            input_ = input_ * masks[:,:1] * masks[:,1:]

        self.loss = self.criterion(input_, warped_)

        return self.loss, warped