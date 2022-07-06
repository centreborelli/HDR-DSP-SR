""" Python script to train option J """
import scipy
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
#import iio
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter

from numpy import mean
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from models import EncoderNet, DecoderNet, FNet#, AddDecoderNet
from shiftandadd import shiftAndAdd, featureAdd, featureWeight

from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, GaussianLayer, BlurLayer
import iio
import os
from torch.autograd import Variable

from registration import generate_data_disk

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lanczos import *



def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass



def flowEstimation(samplesLR, ME, device, gaussian_filter , warping, sr_ratio = 2):
    """
    Compute the optical flows from the other frames to the reference:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """
    #samplesLR = padding(samplesLR)

    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR.view(-1,1,h,w))
    samplesLRblur = samplesLRblur.view(b, num_im, h, w)

    #b, num_im, h, w = samplesLRblur.shape

    samplesLR_0 = samplesLRblur[:,:1,...] #b, 1, h, w
    #if phase == 'training':
    #    samplesLRblur = samplesLRblur[:,1:,...]

    b, num_im, h, w = samplesLR.shape

    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1,1)  #b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  #b*num_im, h, w
    concat = torch.cat((samplesLRblur.unsqueeze(1), samplesLR_0.unsqueeze(1)), axis = 1) #b*(num_im), 2, h, w 
    flow = ME(concat.to(device)) #b*(num_im), 2, h, w 
    #if phase == 'validation':
    flow[::num_im] = 0

    warploss, _ = warping(samplesLRblur.unsqueeze(1).to(device),samplesLR_0.unsqueeze(1).to(device), flow, losstype = 'Detail')

    return flow.reshape(b, num_im, 2, h, w), warploss



def DeepSaaSuperresolve_weighted_base(samplesLR, flow, base, Encoder, Decoder, device, feature_mode, num_features = 64, sr_ratio=2, phase = 'training'):
    """
    samplesLR: b, num_im, h, w
    flow: b*(num_im-1), 2, h, w
    """
    nb_mode = len(feature_mode)

    #base, detail = base_detail_decomp(samplesLR[:,:1], gaussian_filter) #b, 1, h, w

    if phase == 'training':        
        samplesLR = samplesLR[:,1:,...].contiguous() #b, (num_im-1), h, w
        flow = flow[:,1:].contiguous()#.view(-1, 1, 2, h, w)
        base = base[:,1:].contiguous()
    b, num_im, h, w = samplesLR.shape

    #base = base.repeat(1,num_im, 1,1).view(-1,1,h,w)
    #base = warping.warp(base, flow.view(-1,2,h,w)) #b*num_im, 1, h, w

    samplesLR = samplesLR.view(-1,1,h,w)
    base = base.view(-1,1,h,w)

    
    inputEncoder = torch.cat((samplesLR, base), dim = 1)#samplesLR_detail.view(-1, 1, h, w) #b*(num_im-1), 1, h, w
    features = Encoder(inputEncoder) #b * (num_im-1), num_features, h, w
    features = features.view(-1, h, w) # b * num_im-1 *num_features, h, w

    dacc = featureWeight(flow.view(-1,2,h,w),sr_ratio=sr_ratio, device = device)
    flow = flow.contiguous().view(-1, 1, 2, h, w).repeat(1,num_features,1,1,1).view(-1,2, h, w) #b * num_im-1 * num_features, 2, h, w
    dadd = featureAdd(features, flow, sr_ratio=sr_ratio, device = device) #b * num_im * num_features, 2h, 2w

    dadd = dadd.view(b, num_im, num_features, sr_ratio*h, sr_ratio*w)
    dacc = dacc.view(b, num_im, 1, sr_ratio*h, sr_ratio*w)

    SR = torch.empty(b, 1+nb_mode*num_features, sr_ratio*h, sr_ratio*w)
    for i in range(nb_mode):
        if feature_mode[i] == 'Max':
            SR[:, i*num_features:(i+1)*num_features], _ = torch.max(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Std':
            SR[:, i*num_features:(i+1)*num_features] = torch.std(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Avg':
            #dadd = torch.sum(dadd, 1) #b, num_features, sr_ratioh, sr_ratiow
            dacc = torch.sum(dacc, 1)
            dacc[dacc == 0] = 1
            SR[:, i*num_features:(i+1)*num_features] = torch.sum(dadd, 1)/dacc
            SR[:, -1:] = dacc/15.
    SR = Decoder(SR.to(device)) #b, 1, sr_ration*h, sr_ratio*w
    #SR = torch.squeeze(SR, 1)

    return SR


def DeepSaaSuperresolve_weighted(samplesLR, flow, Encoder, Decoder, device, feature_mode, num_features = 64, sr_ratio=2, phase = 'training'):
    """
    samplesLR: b, num_im, h, w
    flow: b*(num_im-1), 2, h, w
    """
    nb_mode = len(feature_mode)


    if phase == 'training':        
        samplesLR = samplesLR[:,1:,...].contiguous() #b, (num_im-1), h, w
        flow = flow[:,1:].contiguous()#.view(-1, 1, 2, h, w)
    b, num_im, h, w = samplesLR.shape


    samplesLR = samplesLR.view(-1,1,h,w)

    
    inputEncoder = samplesLR
    features = Encoder(inputEncoder) #b * (num_im-1), num_features, h, w
    features = features.view(-1, h, w) # b * num_im-1 *num_features, h, w

    dacc = featureWeight(flow.view(-1,2,h,w),sr_ratio=sr_ratio, device = device)
    flow = flow.contiguous().view(-1, 1, 2, h, w).repeat(1,num_features,1,1,1).view(-1,2, h, w) #b * num_im-1 * num_features, 2, h, w
    dadd = featureAdd(features, flow, sr_ratio=sr_ratio, device = device) #b * num_im * num_features, 2h, 2w

    dadd = dadd.view(b, num_im, num_features, sr_ratio*h, sr_ratio*w)
    dacc = dacc.view(b, num_im, 1, sr_ratio*h, sr_ratio*w)

    SR = torch.empty(b, 1+nb_mode*num_features, sr_ratio*h, sr_ratio*w)
    for i in range(nb_mode):
        if feature_mode[i] == 'Max':
            SR[:, i*num_features:(i+1)*num_features], _ = torch.max(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Std':
            SR[:, i*num_features:(i+1)*num_features] = torch.std(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Avg':
            #dadd = torch.sum(dadd, 1) #b, num_features, sr_ratioh, sr_ratiow
            dacc = torch.sum(dacc, 1)
            dacc[dacc == 0] = 1
            SR[:, i*num_features:(i+1)*num_features] = torch.sum(dadd, 1)/dacc
            SR[:, -1:] = dacc/15.
    SR = Decoder(SR.to(device)) #b, 1, sr_ration*h, sr_ratio*w
    #SR = torch.squeeze(SR, 1)

    return SR



class SkySatRealDataset_ME(Dataset):
    def __init__(self, path, augmentation = False,  phase = 'train', normalization = 3400., num_images = 15):
        self.expotime = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(num_images), '{}Ratio.npy'.format(phase)))[...,None,None])
        self.data = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(num_images), '{}LR.npy'.format(phase)))/normalization)

            
        self.len = self.expotime.size()[0]
        self.augmentation = augmentation
        self.num_images = num_images
    def transform(self, data):
        # Random crop
        h, w = data.shape[-2:]
        new_h, new_w = (64,64)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[:, top: top + new_h, left: left + new_w]

        if random.random() < 0.5:
            data = torch.flip(data, [-1])

        # Random vertical flipping
        if random.random() < 0.5:
            data = torch.flip(data, [-2])
            
        if random.random() < 0.5:
            k = random.sample([1,3], 1)[0]
            data = torch.rot90(data, k, [-2, -1])
        
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        expotime = self.expotime[idx]
        if self.augmentation:
            data = self.transform(data)
        return data, expotime

    def __len__(self):
        return self.len


def BicubicWarping(x, flo, device, ds_factor = 2):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if torch.sum(flo*flo) == 0:
        return x[...,::2,::2]
    else:
        B, _, H, W = flo.size()

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(device)
        #grid = grid.cuda()
                #print(grid.shape)
        vgrid = ds_factor*(Variable(grid) + flo)
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(ds_factor*W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(ds_factor*H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)       

        output = nn.functional.grid_sample(x, vgrid,align_corners = True,mode = 'bicubic', padding_mode = 'reflection')
    
        return output


def trainESA(args):  
    criterion = nn.L1Loss()
    seed_everything()    
    train_bs, val_bs, lr_fnet, factor_fnet, patience_fnet, lr_decoder, factor_decoder, patience_decoder, lr_encoder, factor_encoder, patience_encoder, num_epochs, warp_weight, TVflow_weight= args.train_bs, args.val_bs, args.lr_fnet, args.factor_fnet, args.patience_fnet,  args.lr_decoder, args.factor_decoder, args.patience_decoder, args.lr_encoder, args.factor_encoder, args.patience_encoder, args.num_epochs, args.warp_weight, args.TVflow_weight
    num_features, num_blocks = args.num_features, args.num_blocks
    sigma = args.sigma
    sr_ratio = args.sr_ratio
    feature_mode = args.feature_mode
    nb_mode = len(feature_mode)
    print(feature_mode)
    ##################
    #folder_name = 'Test'
    folder_name = 'Real_{}_N2N_FNet_ME_deconv_DetaAtte_W_JS_V_noisy_valvar_time_{}'.format(feature_mode,
        f"{datetime.datetime.now():%m-%d-%H-%M-%S}")



    ################## load Models 
    checkpoint_path = '/mnt/ddisk/nguyen/cnn-sr/TrainHistory/FNet_Real_woSat/time_11-02-16-35-29/checkpoint_1600.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Decoder = DecoderNet(in_dim=1+nb_mode*num_features).float().to(device)


    Encoder = EncoderNet(in_dim=2,conv_dim=64, out_dim=num_features, num_blocks=num_blocks).float().to(device)
    Fnet = FNet().float().to(device)
    Fnet.load_state_dict(checkpoint['state_dictFnet']) 
    

    optimizerFnet = torch.optim.Adam(Fnet.parameters(), lr = lr_fnet)
    optimizerDecoder = torch.optim.Adam(Decoder.parameters(), lr = lr_decoder)
    optimizerEncoder = torch.optim.Adam(Encoder.parameters(), lr = lr_encoder)

    schedulerFnet = torch.optim.lr_scheduler.StepLR(optimizerFnet, step_size=patience_fnet,
                                                 gamma=factor_fnet)
    schedulerDecoder = torch.optim.lr_scheduler.StepLR(optimizerDecoder, step_size=patience_decoder, 
                                                 gamma=factor_decoder)
    schedulerEncoder = torch.optim.lr_scheduler.StepLR(optimizerEncoder, step_size=patience_encoder, 
                                                     gamma=factor_encoder)

    blur_filter_SR = BlurLayer().to(device)

    gaussian_filter = GaussianLayer(sigma=1).to(device)

    TVLoss = TVL1(TVLoss_weight=1)
    warping = WarpedLoss(interpolation = 'bicubicTorch') 
    ##################
    
    Dataset_path = '/mnt/ddisk/nguyen/cnn-sr/datareal_ME_woSat/'
    train_loader = {}
    val_loader = {}
    
    for i in range(4,16):
        transformedDataset = SkySatRealDataset_ME(Dataset_path, augmentation = True, phase = 'train', num_images = i)
        train_loader[str(i)] = torch.utils.data.DataLoader(transformedDataset, batch_size=train_bs, 
                                           num_workers=4, shuffle=True)       
    for i in range(8,9):
        transformedDataset = SkySatRealDataset_ME(Dataset_path, augmentation = False, phase = 'val', num_images = i)
        val_loader[str(i)] = torch.utils.data.DataLoader(transformedDataset, batch_size=val_bs, 
                                           num_workers=1, shuffle=False)
    checkpoint_dir = 'TrainHistory/{}'.format(folder_name)
    safe_mkdir(checkpoint_dir)
    with open(checkpoint_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    ##################
    starttime = time()
    best_score = 100 #val_score
    ##################
    for epoch in range(num_epochs):
        TrainLoss = []
        ValLoss = []
        
        TrainN2NLoss = []
        ValN2NLoss = []

        TrainWarpLoss = []
        TrainTVLoss = []

        ValWarpLoss = []
        ValTVLoss = []

        num_images = random.sample(range(4,16), 1)[0]
        
        n = random.sample(range(num_images,16), 1)[0]

        print('__________________________________________________')
        print('Training epoch {0:3d}'.format(epoch))
        for i, (samplesLR, expotime) in enumerate(train_loader[str(n)]):
            """
            samplesLR, samplesLRblur : b, num_im, h, w
            shifts: b, num_im, 2  
            samplesHR: b, 2h, 2w
            """
            optimizerFnet.zero_grad()
            optimizerDecoder.zero_grad()
            optimizerEncoder.zero_grad()

            idx = random.sample(range(n), num_images)
            #good_image = random.sample(list(torch.where(expotime>1)[0]),1)[0]
            
            samplesLR = samplesLR[:, idx].float().to(device)
  
            b, num_im, h, w = samplesLR.shape
            
            expotime = expotime[:, idx].float().to(device)
            samplesLR = samplesLR/expotime
        
            base, detail = base_detail_decomp(samplesLR, gaussian_filter) #b, 1, h, w 
            
            #######Flow
            flow, trainwarploss = flowEstimation(samplesLR*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device=device) #b*(num_im-1), 2, h, w
            
            c = 5
            traintvloss = TVLoss(flow[...,c:-c,c:-c])

            TrainWarpLoss.append(trainwarploss.data.item())
            TrainTVLoss.append(traintvloss.data.item())

            random_shifts = torch.randint(low=0, high=2, size= (b,1,2,1,1))/2.
            flow = flow - random_shifts.to(device)

            SR1 = DeepSaaSuperresolve_weighted_base(detail, flow=flow, base = samplesLR, Encoder=Encoder, Decoder =Decoder, 
                                      device = device, feature_mode = feature_mode, num_features = num_features, sr_ratio=sr_ratio, phase = 'training')
            
            ################## Register SR            
            SR1 = blur_filter_SR(SR1)
            #SR1 = torch.squeeze(SR1, 1)
            
            #SR1_ds = SR1[...,::2,::2] 
            SR1_ds = BicubicWarping(SR1.view(-1,1,2*h,2*w), flow[:,:1].view(-1,2,h,w), device)
            SR1_ds = SR1_ds.view(b,1,h,w)

            N2Nloss = criterion(SR1_ds[:, :1, c:-c, c:-c], detail[:,:1,c:-c,c:-c])

            trainloss = N2Nloss + warp_weight*trainwarploss+ TVflow_weight* traintvloss
            
            TrainN2NLoss.append(N2Nloss.data.item())
            TrainLoss.append(trainloss.data.item())
           
            trainloss.backward()
            optimizerFnet.step()
            optimizerDecoder.step()
            optimizerEncoder.step()
            
        if epoch <3000:
            print('Train')
            print('{:.5f} = {:.5f} + {} * {:.5f} + {} * {:.5f}'.format(1000*mean(TrainLoss), 1000*mean(TrainN2NLoss), warp_weight, 
                                                                      1000*mean(TrainWarpLoss), TVflow_weight, 1000*mean(TrainTVLoss)))

        Fnet.eval()
        Decoder.eval()
        Encoder.eval()

        with torch.no_grad():
            for n in range(8,9):
                for k, (samplesLR, expotime) in enumerate(val_loader[str(n)]):

                    samplesLR = samplesLR.float().to(device)

                    b, num_im, h, w = samplesLR.shape
                    expotime = expotime.float().to(device)

                    samplesLR = samplesLR/expotime
                    #######Flow
                    flow, valwarploss = flowEstimation(samplesLR*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device=device) #b*(num_im-1), 2, h, w

                    c = 5
                    valtvloss = TVLoss(flow[...,c:-c,c:-c])

                    ValWarpLoss.append(valwarploss.data.item())
                    ValTVLoss.append(valtvloss.data.item())

                    #flow = shift[...,None, None].repeat(1,1,1,h,w)

                    base, detail = base_detail_decomp(samplesLR, gaussian_filter) 
                    SR = DeepSaaSuperresolve_weighted_base(detail, flow=flow, base = samplesLR, Encoder=Encoder, Decoder=Decoder,
                                         device = device, feature_mode= feature_mode, num_features = num_features, sr_ratio=sr_ratio, phase = 'validation')

                    SRb = blur_filter_SR(SR)

                    SR_ds = SRb[...,::2,::2]

                    N2Nloss = criterion(SR_ds[:, :1, c:-c, c:-c], detail[:,:1,c:-c,c:-c])
                    ValN2NLoss.append(N2Nloss.data.item())


                    valloss = N2Nloss + warp_weight*valwarploss+ TVflow_weight* valtvloss


                    #valloss = N2Nloss
                    ValLoss.append(valloss.data.item())
                    
                    if k == 0 and epoch%10==0:
                        SR = SR.detach().cpu().numpy().squeeze()
                        iio.write('Test/SR_{}.tif'.format(epoch), SR)



        if epoch<3000:
            print('Val')
            print('{:.5f} = {:.5f} + {} * {:.5f} + {} * {:.5f}'.format(1000*mean(ValLoss), 1000*mean(ValN2NLoss), warp_weight, 
                                                                      1000*mean(ValWarpLoss), TVflow_weight, 1000*mean(ValTVLoss)))


        
        schedulerFnet.step()
        schedulerDecoder.step()
        schedulerEncoder.step()
        
        if  epoch >= 300 and epoch%100 == 0:
            print('#### Saving Models ... ####')
            print('#### Saving Models ... ####')
            state = {'epoch': epoch + 1,'state_dictDecoder':Decoder.state_dict(),'optimizerDecoder': optimizerDecoder.state_dict(), 'state_dictEncoder':Encoder.state_dict(), 'optimizerEncoder': optimizerEncoder.state_dict(), 'state_dictFnet':Fnet.state_dict(),'optimizerFnet': optimizerFnet.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

    
    print('Execution time = {:.0f}s'.format(time() - starttime))
    return 


def zoombase(LR_base, flow, device, warping, mean = 'geo'):
    b, num_im, h, w = LR_base.shape
    if mean == 'geo':
        LR_base = LR_base.view(-1,1,h,w)
        LR_base = warping.warp(LR_base, -flow.view(-1,2,h,w))
        LR_base = LR_base.view(b,num_im, h, w)
        LR_base = torch.log(LR_base)
        LR_base = torch.mean(LR_base, 1, keepdim = True)
        LR_base = torch.exp(LR_base)
    elif mean == 'ari':
        LR_base = LR_base.view(-1,1,h,w)
        LR_base = warping.warp(LR_base, -flow.view(-1,2,h,w))
        LR_base = LR_base.view(b,num_im, h, w)
        LR_base = torch.mean(LR_base, 1, keepdim = True)
    else:
        LR_base = LR_base[:,:1]
    SR_base = torch.nn.functional.interpolate(LR_base, size = [2*h-1, 2*w-1], mode = 'bilinear', align_corners = True)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,1,2*w-1).to(device)), dim = 2)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,2*h,1).to(device)), dim = 3)
    return SR_base

def zoombase_weighted(LR_base, expotime, flow, device, warping, mean = 'ari'):
    b, num_im, h, w = LR_base.shape
    if mean == 'geo':
        LR_base = LR_base.view(-1,1,h,w)
        LR_base = warping.warp(LR_base, -flow.view(-1,2,h,w))
        LR_base = LR_base.view(b,num_im, h, w)
        LR_base = torch.log(LR_base)
        LR_base = torch.mean(LR_base, 1, keepdim = True)
        LR_base = torch.exp(LR_base)
    elif mean == 'ari':
        LR_base = LR_base.view(-1,1,h,w)
        LR_base = warping.warp(LR_base, -flow.view(-1,2,h,w))
        LR_base = LR_base.view(b,num_im, h, w)
        LR_base = torch.mean(LR_base*expotime, 1, keepdim = True)/torch.mean(expotime, 1, keepdim = True)
    else:
        LR_base = LR_base[:,:1]
    SR_base = torch.nn.functional.interpolate(LR_base, size = [2*h-1, 2*w-1], mode = 'bilinear', align_corners = True)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,1,2*w-1).to(device)), dim = 2)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,2*h,1).to(device)), dim = 3)
    return SR_base

def valid(args):
    seed_everything()
    train_bs, val_bs, lr_fnet, factor_fnet, patience_fnet, lr_decoder, factor_decoder, patience_decoder, lr_encoder, factor_encoder, patience_encoder, num_epochs, warp_weight, TVflow_weight= args.train_bs, args.val_bs, args.lr_fnet, args.factor_fnet, args.patience_fnet,  args.lr_decoder, args.factor_decoder, args.patience_decoder, args.lr_encoder, args.factor_encoder, args.patience_encoder, args.num_epochs, args.warp_weight, args.TVflow_weight
    num_features, num_blocks = args.num_features, args.num_blocks
    sigma = args.sigma
    sr_ratio = args.sr_ratio
    feature_mode = ['Avg', 'Max', 'Std']

    nb_mode = len(feature_mode)
    print(feature_mode)    
    
    ##################
    folder_name = "Real_woBD_['Avg', 'Max', 'Std']_N2N_FNet_ME_deconv_DetaAtte2_W_JS_V_noisy_valvar_time_11-13-15-08-44"#"Real_['Avg', 'Max', 'Std']_N2N_FNet_ME_deconv_DetaAtte2_W_JS_V_noisy_valvar_time_11-03-09-54-35"#"Real_['Avg', 'Max', 'Std']_N2N_FNet_ME_DetaAtte2_W_JS_V_noisy_valvar_time_11-02-17-10-00"
    ################## load Models 
    checkpoint_path = 'TrainHistory/{}/checkpoint_1700.pth.tar'.format(folder_name)
    checkpoint = torch.load(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Decoder = DecoderNet(in_dim=1+(nb_mode)*num_features).float().to(device)

    Decoder.load_state_dict(checkpoint['state_dictDecoder'])

    Encoder = EncoderNet(in_dim=1,conv_dim=64, out_dim=num_features, num_blocks=num_blocks).float().to(device)
    Encoder.load_state_dict(checkpoint['state_dictEncoder'])

    Fnet = FNet().float().to(device)
    Fnet.load_state_dict(checkpoint['state_dictFnet']) 
    print(checkpoint['epoch'])
    
    TVLoss = TVL1(TVLoss_weight=1)
    gaussian_filter = GaussianLayer(sigma=1).to(device)
    warping = WarpedLoss(interpolation = 'bicubicTorch') 
    ##################


    #Test loader
    
    Dataset_path = '/mnt/ddisk/nguyen/cnn-sr/datareal_ME_woSat/'
    test_loader = {}
    
    for i in range(4,16):
        transformedDataset = SkySatRealDataset_ME(Dataset_path, augmentation = False, phase = 'test', num_images = i)
        test_loader[str(i)] = torch.utils.data.DataLoader(transformedDataset, batch_size=val_bs, 
                                           num_workers=1, shuffle=False)

    ##################
    starttime = time()
    ##################

    Fnet.eval()
    Decoder.eval()
    Encoder.eval()

    with torch.no_grad():
        for n in range(4,16):
            image_folder_base = os.path.join('TestME/Real/', folder_name, 'SRbase', str(n))
            safe_mkdir(image_folder_base)
            image_folder_detail = os.path.join('TestME/Real/', folder_name, 'SRdetail', str(n))
            safe_mkdir(image_folder_detail)
            image_folder = os.path.join('TestME/Real/', folder_name, 'SR', str(n))
            safe_mkdir(image_folder)
            #image_LR = 'TestME/Real/LR_ref/'+ str(n)
            #safe_mkdir(image_LR)
            for k, (samplesLR, expotime) in enumerate(test_loader[str(n)]):
                """
                samplesLR, samplesLRblur : b, num_im, h, w
                samplesHR: b, 2h, 2w
                """
                
                idx = list(range(n))
                idx.remove(n//2)
                idx.insert(0, n//2)
                samplesLR = samplesLR[:,idx].float().to(device)

                b, num_im, h, w = samplesLR.shape
                expotime = expotime[:,idx].float().to(device)

                #######Flow
                samplesLR = samplesLR/expotime
                
                flow, _ = flowEstimation(samplesLR*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device=device) #b*(num_im-1), 2, h, w


                base, detail = base_detail_decomp(samplesLR, gaussian_filter) 
                
                SR_detail = DeepSaaSuperresolve_weighted(samplesLR, flow=flow, Encoder=Encoder, Decoder=Decoder,
                                     device = device, feature_mode= feature_mode, num_features = num_features, sr_ratio=sr_ratio, phase = 'validation')
                

                SR_baseAri = zoombase(base, flow, device, warping, mean = 'ari')

                SR_Ari = SR_detail # + SR_baseAri

                c = 10
                SR_baseAri = torch.squeeze(SR_baseAri).detach().cpu().numpy()[c:-c,c:-c]
                SR_detail = torch.squeeze(SR_detail).detach().cpu().numpy()[c:-c,c:-c]

                SR_Ari = torch.squeeze(SR_Ari).detach().cpu().numpy()[c:-c,c:-c]
                LR = torch.squeeze(samplesLR,0).detach().cpu().numpy()[0,c//2:-c//2,c//2:-c//2]
                LR_base = torch.squeeze(base,0).detach().cpu().numpy()[0,c//2:-c//2,c//2:-c//2]
                #LR_normalized = torch.squeeze(samplesLR/expotime,0).detach().cpu().numpy()[:,c//2:-c//2,c//2:-c//2]

                SR_Ari = SR_Ari/np.median(SR_Ari)*np.median(LR)
                SR_baseAri = SR_baseAri/np.median(SR_baseAri)*np.median(LR_base)

                
                #iio.write(os.path.join(image_folder_base, 'SR_base_{:03d}.tif'.format(k)), SR_baseAri)
                #iio.write(os.path.join(image_folder_detail, 'SR_detail_{:03d}.tif'.format(k)), SR_detail)

                iio.write(os.path.join(image_folder, 'SR_{:03d}.tif'.format(k)), SR_Ari)
                #iio.write(os.path.join(image_LR, 'LR_{:03d}.tif'.format(k)), LR)


    return
                

def check(args):
    feature_mode = args.feature_mode
    print(feature_mode)
    print(len(feature_mode))

def main(args):
    """
    Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    torch.cuda.empty_cache()
    #resume_training(args)
    trainESA(args)
    #valid(args)
    #finetune_mire(args)
    #check(args)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-on","--option_name", help="Option id", default='J_selfSR')
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader",type=int, default=10)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader",type=int, default=1)
    parser.add_argument("-lrf", "--lr_fnet", help="Learning rate of fnet",type=float, default=1e-5)
    parser.add_argument("-lre", "--lr_encoder", help="Learning rate of Encoder",type=float, default=1e-4)
    parser.add_argument("-lrd", "--lr_decoder", help="Learning rate of Decoder",type=float, default=1e-4)
    parser.add_argument("-ff",  "--factor_fnet", help="Learning rate decay factor of fnet",type=float, default=0.3)
    parser.add_argument("-fe",  "--factor_encoder", help="Learning rate decay factor of Encoder",type=float, default=0.3)
    parser.add_argument("-fd",  "--factor_decoder", help="Learning rate decay factor of Decoder",type=float, default=0.3)
    parser.add_argument("-pf",  "--patience_fnet", help="Step size for learning rate of fnet",type=int, default=300)
    parser.add_argument("-pe",  "--patience_encoder", help="Step size for learning rate of Encoder",type=int, default=400)
    parser.add_argument("-pd",  "--patience_decoder", help="Step size for learning rate of Decoder",type=int, default=400)
    parser.add_argument("-ne",  "--num_epochs", help="Num_epochs",type=int, default=1800)
    parser.add_argument("-nf",  "--num_features", help="Num of features for each frame", type=int, default=64)
    parser.add_argument("-nb",  "--num_blocks", help="Number of residual blocks in encoder",type=int, default=4)
    parser.add_argument("-ww",  "--warp_weight", help="Weight for the warping loss",type=float, default=3)
    parser.add_argument("-tvw",  "--TVflow_weight", help="Weight for the TV flow loss",type=float, default=0.01)
    parser.add_argument("-s",  "--sigma", help="Std for SR filtering",type=float, default=1)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor",type=int, default=2)
    parser.add_argument('-fm','--feature_mode', nargs='+', help="feature mode (Avg, Max, Std)", default=['Avg', 'Max', 'Std'])


    args = parser.parse_args()

    main(args)


