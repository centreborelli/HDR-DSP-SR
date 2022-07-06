import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np 

def get_neighbours(coords):

    coords_lr = torch.ceil(coords)
    coords_ul = torch.floor(coords)
    ys_upper, xs_left = torch.split(coords_ul, 1, dim = 1)
    ys_lower, xs_right = torch.split(coords_lr, 1, dim = 1)
    coords_ll = torch.cat((ys_lower, xs_left), axis = 1)
    coords_ur = torch.cat((ys_upper, xs_right), axis = 1)
    
    return coords_ul, coords_ur, coords_ll, coords_lr


def coords_unroll(coords, sr_ratio = 2):
    """
    coords : tensor(b,2,h,w)
    """
    b,c,h,w = coords.shape
    assert(c == 2)
    coords_ = coords.view(b,c,-1)
    coords_ = sr_ratio*w*coords_[:,0] + coords_[:,1]
    return coords_

def get_coords(h, w):
    """get coords matrix of x

    # Arguments
        h
        w
    
    # Returns
        coords: (h, w, 2)
    """
    coords = torch.empty(2, h, w, dtype = torch.float)
    coords[0,...] = torch.arange(h)[:, None]
    coords[1,...] = torch.arange(w)

    return coords


def shiftAndAdd(samples, flows, sr_ratio, device):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """
    
    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...]
    
    
    b, h, w = samples.shape
    samples_= samples.reshape(b, -1).type(torch.float32)

    mapping = sr_ratio*(flows_ + get_coords(h,w).to(device))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).to(device)
    neg_diff = (1.0 - diff).type(torch.float32).to(device)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.reshape(b,-1)
    diff_y = diff_y.reshape(b,-1)
    neg_diff_x = neg_diff_x.reshape(b,-1)
    neg_diff_y = neg_diff_y.reshape(b,-1)
    
    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).to(device)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).to(device)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).to(device)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).to(device)
    
    dadd = torch.zeros(b, sr_ratio*sr_ratio*h*w).to(device)
    dacc = torch.zeros(b, sr_ratio*sr_ratio*h*w).to(device)

    dadd = dadd.scatter_add(1, coords_ul, samples_*neg_diff_x*neg_diff_y)
    dacc = dacc.scatter_add(1, coords_ul, neg_diff_x*neg_diff_y)


    dadd = dadd.scatter_add(1, coords_ur, samples_*diff_x*neg_diff_y)
    dacc = dacc.scatter_add(1, coords_ur, diff_x*neg_diff_y)


    dadd = dadd.scatter_add(1, coords_ll, samples_*neg_diff_x*diff_y)
    dacc = dacc.scatter_add(1, coords_ll, neg_diff_x*diff_y)


    dadd = dadd.scatter_add(1, coords_lr, samples_*diff_x*diff_y)
    dacc = dacc.scatter_add(1, coords_lr, diff_x*diff_y)

    return dadd.view(b, h*sr_ratio, sr_ratio*w), dacc.view(b,h*sr_ratio, sr_ratio*w)

def featureAdd(samples, flows, sr_ratio, device):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """

    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...] #b*n, 2, h, w
    b, h, w = samples.shape

    samples_= samples.view(b, -1).type(torch.float32)

    mapping = sr_ratio*(flows_ + get_coords(h,w).to(device))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).to(device)
    neg_diff = (1.0 - diff).type(torch.float32).to(device)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.view(b,-1)
    diff_y = diff_y.view(b,-1)
    neg_diff_x = neg_diff_x.view(b,-1)
    neg_diff_y = neg_diff_y.view(b,-1)

    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).to(device)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).to(device)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).to(device)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).to(device)

    dadd = torch.zeros(b, sr_ratio*sr_ratio*h*w).to(device)

    dadd = dadd.scatter_add(1, coords_ul, samples_*neg_diff_x*neg_diff_y)

    dadd = dadd.scatter_add(1, coords_ur, samples_*diff_x*neg_diff_y)

    dadd = dadd.scatter_add(1, coords_ll, samples_*neg_diff_x*diff_y)

    dadd = dadd.scatter_add(1, coords_lr, samples_*diff_x*diff_y)

    return dadd.view(b, h*sr_ratio, sr_ratio*w)

def featureWeight(flows, sr_ratio, device):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """

    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...] #b*n, 2, h, w

    b, _, h, w = flows.shape

    mapping = sr_ratio*(flows_ + get_coords(h,w).to(device))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).to(device)
    neg_diff = (1.0 - diff).type(torch.float32).to(device)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.view(b,-1)
    diff_y = diff_y.view(b,-1)
    neg_diff_x = neg_diff_x.view(b,-1)
    neg_diff_y = neg_diff_y.view(b,-1)

    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).to(device)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).to(device)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).to(device)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).to(device)

    dacc = torch.zeros(b, sr_ratio*sr_ratio*h*w).to(device)

    dacc = dacc.scatter_add(1, coords_ul, neg_diff_x*neg_diff_y)

    dacc = dacc.scatter_add(1, coords_ur, diff_x*neg_diff_y)

    dacc = dacc.scatter_add(1, coords_ll, neg_diff_x*diff_y)

    dacc = dacc.scatter_add(1, coords_lr, diff_x*diff_y)

    return dacc.view(b,h*sr_ratio, sr_ratio*w)

