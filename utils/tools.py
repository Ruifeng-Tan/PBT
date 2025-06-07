import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import time
from torch import nn
from utils.losses import WeightedRnCLoss
import wandb
plt.switch_backend('agg')
def split_meta_domains(domain_ids, K):
    """
    Splits the domain_ids into meta-train and meta-test domains based on K% test domains.
    
    Args:
        domain_ids (torch.Tensor): A 1D tensor of shape [N] containing domain IDs for each sample.
        K (float): Percentage of domains to allocate to meta-test (0 <= K <= 100).
    
    Returns:
        tuple: Two tensors, the first containing indices of meta-train samples and the second of meta-test samples.
    """
    unique_domains = torch.unique(domain_ids)
    num_unique = unique_domains.size(0)
    # Calculate the number of test domains, ensuring it's an integer
    num_test = int(round(num_unique * K / 100.0))  
    num_test = 1 if num_test == 0 else num_test
    
    # Handle edge cases where K is 0 or 100
    if num_test == 0:
        raise Exception("K cannot be 0, as it would result in no test domains.")
    elif num_test >= num_unique:
        raise Exception("K cannot be 100 or more, as it would result in all domains being test domains.")
    else:
        # Randomly permute the domains and split into test and train
        perm = torch.randperm(num_unique)
        shuffled_domains = unique_domains[perm]
        test_domains = shuffled_domains[:num_test]
    
    # Create masks for test and train indices
    test_mask = torch.isin(domain_ids, test_domains)
    test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
    
    train_mask = ~test_mask
    train_indices = torch.nonzero(train_mask, as_tuple=True)[0]
    
    return train_indices, test_indices


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Percent': trainable_num/total_num}

class Augment_time_series_family(object):
    '''
    This is a set of augmentation for methods for time series
    '''
    def __init__(self, n_holes, mean=0, std=0.02):
        pass

class Downsample_Expand_aug(object):
    '''
    '''
    def __init__(self, rate=0.1):
        pass

class Masking_aug(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
    
    def __call__(self, seq):
        '''
        Params:
            seq: Tensor sequence of size (B, num_var, L)
        '''
        seq = F.dropout(seq, self.drop_rate)
        return seq
    
    
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if epoch <= args.warm_up_epoches:
        # The learning rate is controlled by warmup
        return
    else:
        if args.lradj == 'type1':
            # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
            if epoch > args.least_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                new_lr = args.learning_rate * (0.5 ** ((epoch - args.least_epochs) // 1))
                accelerator.print(f'{args.lradj}| Updating learning rate to {new_lr}')
            else:
                accelerator.print(f'{args.lradj}| Updating learning rate to {args.learning_rate}')
        elif args.lradj == 'constant':
            accelerator.print(f'{args.lradj}| Updating learning rate to {args.learning_rate}')
    
    # if epoch in lr_adjust.keys():
    #     lr = lr_adjust[epoch]
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     if printout:
    #         if accelerator is not None:
    #             accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
    #         else:
    #             print(f'{args.lradj}| Updating learning rate to {lr}')

    # if printout:
    #     if accelerator is not None:
    #         accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
    #     else:
    #         print(f'{args.lradj}| Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, args, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, least_epochs=5):
        self.args = args
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_vali_mae = None
        self.best_test_mae = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.least_epochs = least_epochs

    def __call__(self, epoch, val_loss, vali_mae_loss, test_mae_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            if epoch > self.least_epochs:
                # the early stopping won't count before some epoches are trained
                self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_vali_mae = vali_mae_loss
            # self.best_test_mae = test_mae_loss
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            
            # model = self.accelerator.unwrap_model(model)
            # self.accelerator.save(model.state_dict(), path + '/' + 'checkpoint')
            # self.accelerator.wait_for_everyone()
            # if self.args.use_LoRA:
            #     model.save_pretrained(path, save_adapter=True, save_config=True)
            #     self.accelerator.save_model(model, path)
            # else:
            #     self.accelerator.save_model(model, path)
            self.accelerator.save_model(model, path)
            # self.accelerator.save_state(path)
            #self.accelerator.save(model, path + '/' + 'checkpoint.pth')
            self.accelerator.print(f'The checkpoint is saved in {path}!')
            # self.accelerator.save_state(path + '/')
        else:
            #torch.save(model, path + '/' + 'checkpoint.pth')
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False):
    model.eval()
    total_preds, total_references = [], []
    total_seen_unseen_ids = []
    std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, labels, _,  _, DKP_embeddings, seen_unseen_ids, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids) in enumerate(vali_loader):
            if accelerator is None:
                # use the GPU manually
                cycle_curve_data = cycle_curve_data.to(torch.bfloat16).cuda()
                curve_attn_mask = curve_attn_mask.to(torch.bfloat16).cuda()
                DKP_embeddings = DKP_embeddings.to(torch.bfloat16).cuda()
                cathode_masks = cathode_masks.to(torch.bfloat16).cuda()
                temperature_masks = temperature_masks.to(torch.bfloat16).cuda()
                format_masks = format_masks.to(torch.bfloat16).cuda()
                anode_masks = anode_masks.to(torch.bfloat16).cuda()
                combined_masks = combined_masks.to(torch.bfloat16).cuda()
                labels = labels.cuda()


            # encoder - decoder
            outputs, _, _, _, _, _, _, _ = model(cycle_curve_data, curve_attn_mask, DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks
                                                 , temperature_masks=temperature_masks, format_masks=format_masks, anode_masks=anode_masks,
                                                 combined_masks=combined_masks, ion_type_masks=ion_type_masks, use_aug=False)
            # self.accelerator.wait_for_everyone()
            
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            if accelerator is None:
                all_predictions, all_targets, seen_unseen_ids = transformed_preds, transformed_labels, seen_unseen_ids
            else:
                all_predictions, all_targets, seen_unseen_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, seen_unseen_ids))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            if compute_seen_unseen:
                total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()
                
    total_preds = np.array(total_preds)
    total_references = np.array(total_references)   
    total_seen_unseen_ids = np.array(total_seen_unseen_ids)    
    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha1)
    alpha_acc1 = hit_num / len(total_references) * 100

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha2)
    alpha_acc2 = hit_num / len(total_references) * 100

    if compute_seen_unseen:
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1]
        unseen_references = total_references[total_seen_unseen_ids==0]
        seen_preds = total_preds[total_seen_unseen_ids==1]
        unseen_preds = total_preds[total_seen_unseen_ids==0]

        # MAPE
        if len(seen_preds) > 0:
            seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        else:
            seen_mape = -10000

        if len(unseen_preds) > 0:
            unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
        else:
            unseen_mape = -10000

        # alpha-acc1 
        if len(seen_preds) > 0:
            relative_error = abs(seen_preds - seen_references) / seen_references
            hit_num = sum(relative_error<=args.alpha1)
            seen_alpha_acc1 = hit_num / len(seen_references) * 100
        else:
            seen_alpha_acc1 = -10000

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha1)
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc1 = -10000

        # alpha-acc2
        if len(seen_preds) > 0:
            relative_error = abs(seen_preds - seen_references) / seen_references
            hit_num = sum(relative_error<=args.alpha2)
            seen_alpha_acc2 = hit_num / len(seen_references) * 100
        else:
            seen_alpha_acc2 = -10000

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc2 = -10000

        model.train()
        return  rmse, mae, mape, alpha_acc1, alpha_acc2, unseen_mape, seen_mape, unseen_alpha_acc1, seen_alpha_acc1, unseen_alpha_acc2, seen_alpha_acc2

    model.train()
    return rmse, mae, mape, alpha_acc1, alpha_acc2

def domain_average(total_domain_ids, MAPEs, return_IDs=False):
    assert total_domain_ids.shape[0] == MAPEs.shape[0], "Inputs must have the same length"
    
    device = MAPEs.device
    total_domain_ids = total_domain_ids.to(device)
    
    unique_ids, inverse_indices, counts = torch.unique(total_domain_ids, 
                                                      return_inverse=True, 
                                                      return_counts=True)
    domain_N = unique_ids.shape[0]
    res = torch.zeros(domain_N, dtype=MAPEs.dtype, device=device)
    res.index_add_(0, inverse_indices, MAPEs)
    counts = counts.to(MAPEs.dtype)
    res /= counts  # Compute averages
    
    if return_IDs:
        return np.array(unique_ids), np.array(res)  # Return domain IDs and their average MAPEs
    else:
        return np.array(res)
    
def vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False):
    model.eval()
    total_preds, total_references = [], []
    total_seen_unseen_ids = []
    std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, labels, _,  _, DKP_embeddings, seen_unseen_ids, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids) in enumerate(vali_loader):
            if accelerator is None:
                # use the GPU manually
                cycle_curve_data = cycle_curve_data.to(torch.bfloat16).cuda()
                curve_attn_mask = curve_attn_mask.to(torch.bfloat16).cuda()
                DKP_embeddings = DKP_embeddings.to(torch.bfloat16).cuda()
                cathode_masks = cathode_masks.to(torch.bfloat16).cuda()
                temperature_masks = temperature_masks.to(torch.bfloat16).cuda()
                format_masks = format_masks.to(torch.bfloat16).cuda()
                anode_masks = anode_masks.to(torch.bfloat16).cuda()
                combined_masks = combined_masks.to(torch.bfloat16).cuda()
                labels = labels.cuda()


            # encoder - decoder
            outputs = model(cycle_curve_data, curve_attn_mask, DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks
                                                 , temperature_masks=temperature_masks, format_masks=format_masks, anode_masks=anode_masks,
                                                 combined_masks=combined_masks, ion_type_masks=ion_type_masks, use_aug=False)
            # self.accelerator.wait_for_everyone()
            
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            if accelerator is None:
                all_predictions, all_targets, seen_unseen_ids = transformed_preds, transformed_labels, seen_unseen_ids
            else:
                all_predictions, all_targets, seen_unseen_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, seen_unseen_ids))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            if compute_seen_unseen:
                total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()
                
    total_preds = np.array(total_preds)
    total_references = np.array(total_references)   
    total_seen_unseen_ids = np.array(total_seen_unseen_ids)    
    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha1)
    alpha_acc1 = hit_num / len(total_references) * 100

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha2)
    alpha_acc2 = hit_num / len(total_references) * 100

    if compute_seen_unseen:
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1]
        unseen_references = total_references[total_seen_unseen_ids==0]
        seen_preds = total_preds[total_seen_unseen_ids==1]
        unseen_preds = total_preds[total_seen_unseen_ids==0]

        # MAPE
        if len(seen_preds) > 0:
            seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        else:
            seen_mape = -10000

        if len(unseen_preds) > 0:
            unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
        else:
            unseen_mape = -10000

        # alpha-acc1 
        if len(seen_preds) > 0:
            relative_error = abs(seen_preds - seen_references) / seen_references
            hit_num = sum(relative_error<=args.alpha1)
            seen_alpha_acc1 = hit_num / len(seen_references) * 100
        else:
            seen_alpha_acc1 = -10000

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha1)
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc1 = -10000

        # alpha-acc2
        if len(seen_preds) > 0:
            relative_error = abs(seen_preds - seen_references) / seen_references
            hit_num = sum(relative_error<=args.alpha2)
            seen_alpha_acc2 = hit_num / len(seen_references) * 100
        else:
            seen_alpha_acc2 = -10000

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc2 = -10000

        model.train()
        return  rmse, mae, mape, alpha_acc1, alpha_acc2, unseen_mape, seen_mape, unseen_alpha_acc1, seen_alpha_acc1, unseen_alpha_acc2, seen_alpha_acc2

    model.train()
    return rmse, mae, mape, alpha_acc1, alpha_acc2