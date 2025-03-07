import torch
import numpy as np
import copy

def augmentation(augment_time):
    if augment_time == 'batch':
        return BatchAugmentation()
    elif augment_time == 'dataset':
        return DatasetAugmentation()


class BatchAugmentation_battery_revised():
    def __init__(self, aug_rate=0.5, cut_rate=0.5, holes=10, length=5, std=0.02):
        self.cut_rate = cut_rate
        self.aug_rate = aug_rate
        self.cutout_aug = Cutout_jitter_aug(holes, length, std=std)

    def freq_mask(self, x, rate=0.25, dim=1):
        '''
        x: [B*L, charge_discharg_len]
        '''
        xy = x
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.ones(xy_f.shape, dtype=x.dtype, device=x.device)
        m = m.uniform_(0, 1) < rate
        # m = torch.cuda.FloatTensor(xy_f.shape).uniform_() 
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def batch_aug(self, x):
        '''
        Augment the current and voltage records
        Args:
            x (Tensor): shape [B*L, num_var, charge_discharg_len]
        '''
        x = torch.from_numpy(x)
        aug_x = copy.deepcopy(x)
        raw_x = copy.deepcopy(x)
        N, num_var = x.shape[0], x.shape[1]
        voltage_records = x[:,0,:]
        current_records = x[:,1,:]
        capacity_records = x[:,-1,:].unsqueeze(1)
        
        # cutoff-jitter
        cut_aug_voltage = self.cutout_aug(voltage_records.unsqueeze(1))
        cut_aug_current = self.cutout_aug(current_records.unsqueeze(1))

        cut_aug_x = torch.cat([cut_aug_voltage, cut_aug_current, capacity_records], dim=1)

        # Frequency masking
        freqmask_aug_voltage = self.freq_mask(voltage_records)
        freqmask_aug_current = self.freq_mask(current_records)

        freqmask_aug_voltage = freqmask_aug_voltage.unsqueeze(1)
        freqmask_aug_current = freqmask_aug_current.unsqueeze(1)

        freqmask_aug_x = torch.cat([freqmask_aug_voltage, freqmask_aug_current, capacity_records], dim=1)

        # m = torch.ones((N,1,1), dtype=x.dtype, device=x.device)
        # m = m.uniform_(0, 1) < self.cut_rate # set True to use cut_aug
        # m = m.expand_as(aug_x)

        # aug_x = torch.where(m, cut_aug_x, freqmask_aug_x) # randomly use frequency mask and cutoff_jitter

        # m = torch.ones((N,1,1), dtype=x.dtype, device=x.device)
        # m = m.uniform_(0, 1) < self.aug_rate # set True to use cut_aug
        # m = m.expand_as(aug_x)

        # aug_x = torch.where(m, aug_x, raw_x) # only a portion of cycles are replaced by augmented data.
        # aug_x = aug_x.cpu().numpy()
        return cut_aug_x, freqmask_aug_x
    
class BatchAugmentation_battery():
    def __init__(self, cut_rate=0.5, holes=10, length=5, std=0.02):
        self.cut_rate = cut_rate
        self.cutout_aug = Cutout_jitter_aug(holes, length, std=std)

    def freq_mask(self, x, rate=0.25, dim=1):
        xy = x
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.ones(xy_f.shape, dtype=x.dtype, device=x.device)
        m = m.uniform_(0, 1) < rate
        # m = torch.cuda.FloatTensor(xy_f.shape).uniform_() 
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def batch_aug(self, x):
        '''
        Augment the current and voltage records
        Args:
            x (Tensor): shape [B*L, num_var, charge_discharg_len]
        '''
        aug_x = x.clone()
        N, num_var = x.shape[0], x.shape[1]
        voltage_records = x[:,0,:]
        current_records = x[:,1,:]
        capacity_records = x[:,-1,:].unsqueeze(1)
        
        # cutoff-jitter
        cut_aug_voltage = self.cutout_aug(voltage_records.unsqueeze(1))
        cut_aug_current = self.cutout_aug(current_records.unsqueeze(1))

        cut_aug_x = torch.cat([cut_aug_voltage, cut_aug_current, capacity_records], dim=1)

        # Frequency masking
        freqmask_aug_voltage = self.freq_mask(voltage_records)
        freqmask_aug_current = self.freq_mask(current_records)

        freqmask_aug_voltage = freqmask_aug_voltage.unsqueeze(1)
        freqmask_aug_current = freqmask_aug_current.unsqueeze(1)

        freqmask_aug_x = torch.cat([freqmask_aug_voltage, freqmask_aug_current, capacity_records], dim=1)

        m = torch.ones((N,1,1), dtype=x.dtype, device=x.device)
        m = m.uniform_(0, 1) < self.cut_rate # set True to use cut_aug
        m = m.expand_as(aug_x)

        aug_x = torch.where(m, cut_aug_x, freqmask_aug_x)

        return aug_x




class Cutout_jitter_aug(object):
    """
    Randomly cutout some holes of the time series and then replace the
    holes with noised parts.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, mean=0, std=0.01):
        self.n_holes = n_holes
        self.length = length
        self.mean = mean
        self.std = std

    def __call__(self, seq):
        """
        Args:
            seq (Tensor): Tensor sequence of size (B, num_var, L).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        seq = seq.transpose(1,2) # [B, L, num_var]
        L = seq.size(1)

        mask = torch.ones_like(seq[0]) # [L, num_var]
        noise_ratio = torch.normal(self.mean, self.std,size=(seq.shape[0], seq.shape[1], seq.shape[2]), device=seq.device)
        noise_ratio = torch.clip(noise_ratio, min=self.mean-3*self.std, max=self.mean+3*self.std)
        noisy_seq = seq + seq*noise_ratio
        for n in range(self.n_holes):
            y = np.random.randint(L)

            y1 = np.clip(y - self.length // 2, 0, L)
            y2 = np.clip(y + self.length // 2, 0, L)

            mask[y1: y2, :] = 0.

        mask = mask.expand_as(seq)
        seq = seq * mask + (1-mask) * noisy_seq

        return seq.transpose(1,2)
    
class BatchAugmentation():
    def __init__(self):
        pass

    # def freq_mask(self,x, y, rate=0.5, dim=1):
    #     xy = torch.cat([x,y],dim=1)
    #     xy_f = torch.fft.rfft(xy,dim=dim)
    #     m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
    #     freal = xy_f.real.masked_fill(m,0)
    #     fimag = xy_f.imag.masked_fill(m,0)
    #     xy_f = torch.complex(freal,fimag)
    #     xy = torch.fft.irfft(xy_f,dim=dim)
    #     return xy

    def flipping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        # print("flip")
        idxs = np.arange(xy.shape[1])
        idxs = list(idxs)[::-1]
        xy = xy[:,idxs,:]
        return xy

    def warping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        new_xy = torch.zeros_like(xy)
        for i in range(new_xy.shape[1]//2):
            new_xy[:,i*2,:] = xy[:,i + xy.shape[1]//2,:]
            new_xy[:,i*2 + 1,:] = xy[:,i + xy.shape[1]//2,:]
        return xy

    def noise(self,x,y,rate=0.05):
        xy = torch.cat([x,y],dim=1)
        noise_xy = (torch.rand(xy.shape)-0.5) * 0.1
        xy = xy + noise_xy.cuda()
        return xy

    def noise_input(self,x,y,rate=0.05):
        noise = (torch.rand(x.shape)-0.5) * 0.1
        x = x + noise.cuda()
        xy = torch.cat([x,y],dim=1)
        return xy

    def masking(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = np.arange(xy.shape[1])
        np.random.shuffle(b_idx)
        crop_num = int(xy.shape[1]*0.5)
        xy[:,b_idx[:crop_num],:] = 0
        return xy

    def masking_seg(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = int(np.random.rand(1)*xy.shape[1]//2)
        xy[:,b_idx:b_idx+xy.shape[1]//2,:] = 0
        return xy

    def freq_mask(self,x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def freq_mask(self,x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=dim)
        xy_f = torch.fft.rfft(xy,dim=dim)
        
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2,y2],dim=dim)
        xy2_f = torch.fft.rfft(xy2,dim=dim)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

class DatasetAugmentation():
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)

        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate

        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)

        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)
        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=0, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        

        x2, y2 = torch.from_numpy(x2), torch.from_numpy(y2)
        xy2 = torch.cat([x2,y2],dim=0)
        xy2_f = torch.fft.rfft(xy2,dim=0)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=0)
        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y

if __name__=="__main__":
    aug_helper = BatchAugmentation_battery_revised()
    x = np.random.rand(2, 3, 100)
    original_x = copy.deepcopy(x)
    x_aug = aug_helper.batch_aug(x)
    
    if np.all(original_x==x):
        print('No problem')
    if np.any(original_x!=x_aug):
        print('Aug')