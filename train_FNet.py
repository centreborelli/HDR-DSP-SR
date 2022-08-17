import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import Dataset, DataLoader

import scipy
import os
import iio
import datetime
from tensorboardX import SummaryWriter

from models import FNet
from warpingOperator import *

from torchvision.transforms import GaussianBlur


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


class FNetDataset(Dataset):
    def __init__(self, path, augmentation = False,  train=True, normalization = 1000., num_images = 15):
        if train:
            self.expotime = torch.from_numpy(np.load(os.path.join(path, 'train', str(num_images), 'trainRatio.npy'))[...,None,None])
            self.data = torch.from_numpy(np.load(os.path.join(path, 'train', str(num_images), 'trainLR.npy'))/normalization)
        else:
            self.expotime = torch.from_numpy(np.load(os.path.join(path, 'val', str(num_images), 'valRatio.npy'))[...,None,None])
            self.data = torch.from_numpy(np.load(os.path.join(path, 'val', str(num_images), 'valLR.npy'))/normalization)
        self.len = self.expotime.size()[0]
        self.augmentation = augmentation
        self.num_images = num_images
    def transform(self, data, expotime):
        if self.augmentation:
            i = 0
            j = random.sample(range(1,self.num_images), 1)[0]
            
            # Random crop
            h, w = data.shape[-2:]
            new_h, new_w = (80,80)

            top = np.random.randint(10, h-10 - new_h)
            left = np.random.randint(10, w-10 - new_w)

            data = data[:, top: top + new_h, left: left + new_w]
            
            if random.random() < 0.5:
                data = torch.flip(data, [-1])

            # Random vertical flipping
            if random.random() < 0.5:
                data = torch.flip(data, [-2])

        else:
            i, j = 0, 1
            
        sample1 = data[i:i+1,:,:]/expotime[i:i+1]
        sample2 = data[j:j+1,:,:]/expotime[j:j+1]
        data = torch.cat((sample1, sample2), axis = 0)
        
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        expotime = self.expotime[idx]
        
        data = self.transform(data, expotime)
        return data

    def __len__(self):
        return self.len



from tqdm import tqdm
from statistics import mean 
import time
p =1 
interpolation = 'bicubicTorch'
lr = 1e-4
factor = 0.3
patience = 200
TVLoss_weight = 0.003 #0.1 and 0.003 work 0.01 and 0.03 doesn't 
train_bs = 32
val_bs = 20
#criterion = nn.L1Loss()
epochs = 5000
warping = WarpedLoss(p, interpolation = interpolation)
TVLoss = TVL1(TVLoss_weight=1)
losstype = 'Detail'
error = 0
def train():
    seed_everything() 
    folder_name = 'FNet_time_{}'.format(
        f"{datetime.datetime.now():%m-%d-%H-%M-%S}")
    logdir = os.path.join('runs', folder_name)

    tb = SummaryWriter(logdir)
    
    ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Fnet = FNet().float().to(device)
    # Fnet.load_state_dict(torch.load('pretrained_Fnet.pth.tar')['state_dictFnet']) 
    
    optimizer = torch.optim.Adam(Fnet.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                            mode='min', factor=factor, patience=patience,
                            verbose=True)
    
    gaussian_filter = GaussianBlur(11, sigma=1).to(device)

    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')

    #Train loader
    train_loader = {}
    Dataset_path = 'SkySat_ME_noSaturation/'
    
    for i in range(6,16):
        transformedDataset = FNetDataset(Dataset_path, augmentation = True, train=True, num_images = i)
        train_loader[str(i)] = torch.utils.data.DataLoader(transformedDataset, batch_size=train_bs, 
                                           num_workers=4, shuffle=True)
        
    
    transformedDataset = FNetDataset(Dataset_path, augmentation = False, train=False, num_images = 9)

    val_loader = torch.utils.data.DataLoader(transformedDataset, batch_size=val_bs, 
                                           num_workers=2, shuffle=False)
    
    bestscore = 100
    besterror = 1
    checkpoint_dir = os.path.join('TrainHistory/', folder_name)
    safe_mkdir(checkpoint_dir)
    
    
    starttime = time.time()
    for epoch in range(epochs):
        TrainLoss = []
        TrainWarpLoss = []
        TrainTVLoss = []
        ValLoss = []
        ValWarpLoss = []
        ValTVLoss = []
        
        num_images = random.sample(range(6,16), 1)[0]

        for i, (coupleImages) in enumerate(train_loader[str(num_images)]):
            """
            coupleImages: b, 2, h, w
            shift: b, 2
            """
            optimizer.zero_grad()
            
            coupleImages = coupleImages.float().to(device)
            b, _, h, w = coupleImages.shape
            
            coupleImagesblur = gaussian_filter(coupleImages)
            
            flow = Fnet(coupleImagesblur)
            
            trainwarploss, _ = warping(coupleImagesblur[:,:1], coupleImagesblur[:,1:], flow, losstype)
            
            
            traintvloss = TVLoss(flow[...,2:-2,2:-2])
            trainloss = trainwarploss + TVLoss_weight*traintvloss   
            
            TrainLoss.append(trainloss.data.item())
            TrainWarpLoss.append(trainwarploss.data.item())
            TrainTVLoss.append(traintvloss.data.item())
            
            trainloss.backward()
            optimizer.step()
            
            del coupleImages, coupleImagesblur, flow

        tb.add_scalar('Train/WarpLoss', mean(TrainWarpLoss), epoch)
        tb.add_scalar('Train/TVFlowLoss', mean(TrainTVLoss), epoch)
        
        Fnet.eval()
        with torch.no_grad():
            for i, (coupleImages) in enumerate(val_loader):
                """
                coupleImages: b, 2, h, w
                shift: b, 2
                """
            
                coupleImages = coupleImages.float().to(device)
                b, _, h, w = coupleImages.shape
            
                coupleImagesblur = gaussian_filter(coupleImages)
            
                flow = Fnet(coupleImagesblur)
            
                valwarploss, _ = warping(coupleImagesblur[:,:1], coupleImagesblur[:,1:], flow, losstype)
                
                valtvloss = TVLoss(flow[...,2:-2,2:-2])
                valloss = valwarploss + TVLoss_weight*valtvloss   
            
                ValLoss.append(valloss.data.item())
                ValWarpLoss.append(valwarploss.data.item())
                ValTVLoss.append(valtvloss.data.item())        

                del coupleImages, coupleImagesblur, flow

        scheduler.step(mean(TrainLoss))

        tb.add_scalar('Val/WarpLoss', mean(ValWarpLoss), epoch)
        tb.add_scalar('Val/TVFlowLoss', mean(ValTVLoss), epoch)

        if  epoch >= 100 and mean(ValLoss)< bestscore:
            print('############## Saving Models ... ##############')
            print('Epoch {:04d}, old_score = {:.5f}, new_score = {:.5f}'.format(epoch, bestscore, mean(ValLoss)))
            state = {'epoch': epoch + 1, 'state_dictFnet': Fnet.state_dict(), 'optimizerFnet': optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
            bestscore = mean(ValLoss)
        if epoch >= 1000 and epoch%200 == 0:
            print('############## Saving Models ... ##############')
            state = {'epoch': epoch + 1, 'state_dictFnet': Fnet.state_dict(), 'optimizerFnet': optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch)))
            
        if epoch%5 == 0:    
            print('***************************** Epoch {}: ****************************'.format(epoch))
            print('____Train / TrainLoader:____')
            print('x100 TrainLoss = WarpLoss + TVFlow: {:.5f} = {:.5f} + {}*{:.5f}'.format(
                100*mean(TrainLoss), 100*mean(TrainWarpLoss), TVLoss_weight, 100*mean(TrainTVLoss)))
            print('')
            print('____Evaluation / ValLoader:____')
            print('x100 ValLoss   = WarpLoss + TVFlow: {:.5f} = {:.5f} + {}*{:.5f}'.format(
                100*mean(ValLoss), 100*mean(ValWarpLoss), TVLoss_weight, 100*mean(ValTVLoss)))

    tb.close()
    
    print('Execution time = {:.0f}s'.format(time.time() - starttime))
    return 


train()