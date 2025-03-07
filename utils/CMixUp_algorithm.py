'''
Here code is revised from https://github.com/huaxiuyao/C-Mixup/blob/main/src/algorithm.py
'''
import numpy as np
import copy
# import ipdb
import torch
import torch.nn as nn
import time
from torch.optim import Adam
from sklearn.neighbors import KernelDensity
import torch.nn.functional as F

def get_similarity(args, x):
    '''
    x: [N, D]
    '''
    sigma = args.kde_bandwidth
    distance_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1) # [N, N]
    distance_matrix = - torch.pow(distance_matrix, 2) / (2*sigma*sigma) # - ||xi-xj||_2^2 / 2\sigma^2, [N, N]
    sim_score = F.softmax(distance_matrix, dim=1) # [N, N]

    return sim_score

def get_mixup_sample_rate(args, data_packet, use_kde = False):
    
    mix_idx = []
    y_list = data_packet['y_train'] 
    data_list = y_list
    # is_np = isinstance(y_list,np.ndarray)
    # if is_np:
    #     data_list = torch.tensor(y_list, dtype=torch.float32)
    # else:
    #     data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if args.mixtype == 'kde' or use_kde: # kde
            data_i = data_list[i]
            data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D
            
                
            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]
        
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    return mix_idx


def get_batch_kde_mixup_idx(args, Batch_X, Batch_Y):
    assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(args, Batch_packet, use_kde=True) # batch -> kde
        # print(f'Batch_rate[0][:20] = {Batch_rate[0][:20]}')
    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx]) 
            for sel_idx in np.arange(Batch_X.shape[0]//2)]
    return idx2

def get_batch_kde_mixup_batch(args, Batch_X1, Batch_X2, Batch_Y1, Batch_Y2):
    Batch_X = torch.cat([Batch_X1, Batch_X2], dim = 0)
    Batch_Y = torch.cat([Batch_Y1, Batch_Y2], dim = 0)

    idx2 = get_batch_kde_mixup_idx(args,Batch_X,Batch_Y)

    New_Batch_X2 = Batch_X[idx2]
    New_Batch_Y2 = Batch_Y[idx2]
    return New_Batch_X2, New_Batch_Y2