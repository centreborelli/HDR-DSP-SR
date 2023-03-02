""" Python script to train option J """
import numpy as np
import os
import json
import argparse
#import iio
from tqdm import tqdm
from time import time

from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from models import EncoderNet, DecoderNet, FNet
from shiftandadd import shiftAndAdd, featureAdd, featureWeight
from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, GaussianLayer
import iio
import os



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
    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR.view(-1,1,h,w))
    samplesLRblur = samplesLRblur.view(b, num_im, h, w)

    samplesLR_0 = samplesLRblur[:,:1,...] #b, 1, h, w

    b, num_im, h, w = samplesLR.shape

    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1,1)  #b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  #b*num_im, h, w
    concat = torch.cat((samplesLRblur.unsqueeze(1), samplesLR_0.unsqueeze(1)), axis = 1) #b*(num_im), 2, h, w 
    flow = ME(concat.to(device)) #b*(num_im), 2, h, w 
    flow[::num_im] = 0

    warploss, _ = warping(samplesLRblur.unsqueeze(1).to(device),samplesLR_0.unsqueeze(1).to(device), flow, losstype = 'Detail')

    return flow.reshape(b, num_im, 2, h, w), warploss


def DeepSaaSuperresolve_weighted(samplesLR, flow, Encoder, Decoder, device, feature_mode, num_features = 64, sr_ratio=2, phase = 'validation'):
    """
    samplesLR: b, num_im, h, w
    flow: b*(num_im-1), 2, h, w
    """
    b, num_im, h, w = samplesLR.shape
    nb_mode = len(feature_mode)


    inputEncoder = samplesLR 
    inputEncoder = inputEncoder.contiguous().view(-1, 1, h, w) 
    features = Encoder(inputEncoder) 
    features = features.view(-1, h, w) 
    dacc = featureWeight(flow.view(-1,2,h,w),sr_ratio=sr_ratio, device = device)

    flow = flow.view(-1, 1, 2, h, w).repeat(1,num_features,1,1,1).view(-1,2, h, w) 
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
            dacc = torch.sum(dacc, 1)
            dacc[dacc == 0] = 1
            SR[:, i*num_features:(i+1)*num_features] = torch.sum(dadd, 1)/dacc
            SR[:, -1:] = dacc/15.


    SR = Decoder(SR.to(device)) #b, 1, sr_ration*h, sr_ratio*w
    #SR = torch.squeeze(SR, 1)

    return SR



class SkySatSyntheticDataset_forTesting(Dataset):
    def __init__(self, data_path, dataHR_path, shift_path, expotime_path, normalize = 3400.):
        self.data = torch.from_numpy(np.load(data_path)/normalize)       #len, num_im, h, w (h = hcrop, w = wcrop)        
        self.dataHR = torch.from_numpy(np.load(dataHR_path)/normalize)       #len, num_im, h, w (h = hcrop, w = wcrop)                
        self.shift = torch.from_numpy(np.load(shift_path))       #len, num_im, h, w (h = hcrop, w = wcrop)
        self.expotime = torch.from_numpy(np.load(expotime_path))       #len, num_im, h, w (h = hcrop, w = wcrop)        

        self.len = self.data.size()[0]

    
    def __getitem__(self, idx):
        data = self.data[idx]
        dataHR = self.dataHR[idx]
        shift = self.shift[idx]
        expotime = self.expotime[idx]

        return data, dataHR, shift, expotime

    def __len__(self):
        return self.len


def psnr(SR, HR, peak = 1):
    assert(SR.shape == HR.shape)
    mse = np.mean((HR-SR)**2)
    return 10*np.log10(peak**2/mse)



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
    ################## load Models 
    checkpoint_path = "syntheticPretrainedModel/Model_SE.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location = torch.device("cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Decoder = DecoderNet(in_dim=1+(nb_mode)*num_features).float().to(device)

    Decoder.load_state_dict(checkpoint['state_dictDecoder'])

    Encoder = EncoderNet(in_dim=1,conv_dim=64, out_dim=num_features, num_blocks=num_blocks).float().to(device)
    Encoder.load_state_dict(checkpoint['state_dictEncoder'])

    Fnet = FNet().float().to(device)
    Fnet.load_state_dict(checkpoint['state_dictFnet']) 
    print(checkpoint['epoch'])
    gaussian_filter = GaussianLayer(sigma=1).to(device)
    warping = WarpedLoss(interpolation = 'bicubicTorch')
    ##################


    #Test loader
    
    path = #Path to data '/mnt/cdisk/nguyen/cnn-sr/datasynL1B_ME'

    
    DatasetHR_path = os.path.join(path, 'testHR_ME.npy')
    Dataset_path =  os.path.join(path, 'testLR_ME.npy')  #These clean LRs are added noise (fixed by the seed) below

    Shift_path = os.path.join(path, 'testshift_ME.npy')
    expotime_path = os.path.join(path,'testcoeff_ME.npy')

    
    transformedDataset = SkySatSyntheticDataset_forTesting(Dataset_path,DatasetHR_path,Shift_path, expotime_path)

    test_loader = torch.utils.data.DataLoader(transformedDataset, batch_size=1,
                                           num_workers=1, shuffle=False)

    ##################
    starttime = time()
    ##################
    
    test_numim = np.load(os.path.join(path,'test_numim.npy'))
    RegisError_Val = []
    with torch.no_grad():
        Fnet.eval()
        Encoder.eval()
        Decoder.eval()
        image_folder = os.path.join('TestME/ablation', folder_name)
        safe_mkdir(image_folder)

        RMSE = []
        PSNR = []
        SSIM = []

        num_images = "variable"
        PSNRARI = []
        for k, (samplesLR, samplesHR, shift, expotime) in enumerate(test_loader):
                #for k, (samplesLR, samplesHR, shift) in enumerate(test_loader):
                """
                samplesLR, samplesLRblur : b, num_im, h, w
                shifts: b, num_im, 2  
                samplesHR: b, 2h, 2w
                """

                idx = list(range(test_numim[k]))
                samplesLR = samplesLR[:, idx].float().to(device)
                b, num_im, h, w = samplesLR.shape
                shift = shift[:, idx].float().to(device)           
                
                expotime = expotime[:,idx].float().to(device)
                samplesLR = samplesLR/expotime     
                
                ##### Add Noise
                a, c = 0.1187/3400, 12.0497/(3400*3400)
                samplesLR = samplesLR + torch.randn_like(samplesLR)*torch.sqrt(a*samplesLR + c)      #This noise is fixed by the seed
                

                flow, valwarploss = flowEstimation(samplesLR*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device=device) #b*(num_im-1), 2, h, w

                metric = torch.norm(flow-shift[...,None, None], p = 1)/(b*h*w*num_im*2)
                RegisError_Val.append(metric.data.item())
                

                SR = DeepSaaSuperresolve_weighted(samplesLR, flow=flow, Encoder=Encoder, Decoder=Decoder,
                                      device = device, feature_mode= feature_mode, num_features = num_features, sr_ratio=sr_ratio, phase = 'validation')


                c = 4
                SR = torch.squeeze(SR).detach().cpu().numpy()[c:-c,c:-c]

                HR = samplesHR.squeeze().detach().cpu().numpy()[c:-c,c:-c]

                SR = SR/np.sum(SR)*np.sum(HR)

                PsnrAri = psnr(SR, HR)

                PSNRARI.append(PsnrAri)



    f = open(os.path.join(image_folder, "metric.txt"), "a")
    f.write("Test SE, Epoch {:04d}, num_images = {}, PSNR = {:.2f} \n".format(checkpoint['epoch']-1, num_images, np.mean(PSNRARI)))
    f.write("Registration error {:.3f} \n".format(np.mean(RegisError_Val)))
    #f.write(len(test_loader))
    f.write("----------------------------------------------- \n")



    return
                


def main(args):
    """
    Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    torch.cuda.empty_cache()

    valid(args)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-on","--option_name", help="Option id", default='J_selfSR')
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader",type=int, default=7)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader",type=int, default=1)
    parser.add_argument("-lrf", "--lr_fnet", help="Learning rate of fnet",type=float, default=2e-5)
    parser.add_argument("-lre", "--lr_encoder", help="Learning rate of Encoder",type=float, default=1e-4)
    parser.add_argument("-lrd", "--lr_decoder", help="Learning rate of Decoder",type=float, default=1e-4)
    parser.add_argument("-ff",  "--factor_fnet", help="Learning rate decay factor of fnet",type=float, default=0.3)
    parser.add_argument("-fe",  "--factor_encoder", help="Learning rate decay factor of Encoder",type=float, default=0.3)
    parser.add_argument("-fd",  "--factor_decoder", help="Learning rate decay factor of Decoder",type=float, default=0.3)
    parser.add_argument("-pf",  "--patience_fnet", help="Step size for learning rate of fnet",type=int, default=30)
    parser.add_argument("-pe",  "--patience_encoder", help="Step size for learning rate of Encoder",type=int, default=200)
    parser.add_argument("-pd",  "--patience_decoder", help="Step size for learning rate of Decoder",type=int, default=200)
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


