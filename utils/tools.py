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
from utils.losses import Battery_life_CL_loss
import wandb
plt.switch_backend('agg')
def is_training_label_model(epoch, args):
    if (epoch+1) <= args.least_epochs or epoch % 2 == 0:
        return True
    else:
        return False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Percent': trainable_num/total_num}

def train_model_course(args, path, model, model_optim, scheduler, accelerator, train_data, train_loader, vali_data, vali_loader, test_data, test_loader, train_epoches):
    time_now = time.time()
    train_steps = len(train_loader)
    criterion = nn.MSELoss() 
    cl_criterion = Battery_life_CL_loss(args.pos_threshold, args.neg_threshold)
    
    best_vali_loss = float('inf')
    for epoch in range(train_epoches):
        mae_metric = evaluate.load('./utils/mae')
        mape_metric = evaluate.load('./utils/mape')
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_cl_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        for i, (cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class) in enumerate(train_loader):
            with accelerator.accumulate(model):
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                iter_count += 1
                
                cycle_curve_data = cycle_curve_data.float()
                curve_attn_mask = curve_attn_mask.float() # [B, L]
                labels = labels.float()
                input_ids = input_ids.int()
                attention_mask = attention_mask.int()
                

                # encoder - decoder
                outputs, features = model(cycle_curve_data, curve_attn_mask, input_ids=input_ids, attention_mask=attention_mask, contrastive_learning=args.use_contrastive_learning)
                
                if args.use_contrastive_learning:
                    labels = torch.cat([labels, labels], dim=0)
                    cut_off = labels.shape[0]//2
                else:
                    cut_off = labels.shape[0]
                    
                if args.loss == 'MSE':
                    loss = criterion(outputs, labels)
                if args.loss == 'MAPE':
                    tmp_outputs = outputs * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                    
                label_loss = loss.detach().float()
                
                if args.use_contrastive_learning:
                    cl_loss = cl_criterion(features, labels[:cut_off])
                    cl_loss = cl_loss
                    print_cl_loss = cl_loss.detach().float()
                    loss = loss + args.cl_loss_weight * cl_loss
                
                
                print_loss = loss.detach().float()
                
                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                
                mae_metric.add_batch(
                    predictions = transformed_preds,
                    references = transformed_labels
                )
                mape_metric.add_batch(
                    predictions = transformed_preds,
                    references = transformed_labels
                )
                accelerator.backward(loss)
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                model_optim.zero_grad()
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        train_mae_res = mae_metric.compute()
        train_mape_res = mape_metric.compute()
        train_mae = train_mae_res['mae']
        train_mape = train_mape_res['mape']
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_loss, vali_mae_loss, vali_mape = vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion)
        test_loss, test_mae_loss, test_mape = vali_batteryLifeLLM(args, accelerator, model, test_data, test_loader, criterion)

        if args.loss == 'MAPE':
            vali_loss = vali_mape
        elif args.loss == 'MSE':
            pass

        test_rmse, vali_rmse = np.sqrt(test_loss), np.sqrt(vali_loss)
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            
        train_loss = total_loss / len(train_loader)
        total_cl_loss = total_cl_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train MAE: {train_mae:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_MAE": vali_mae_loss, "vali_RMSE":vali_rmse, "test_MAE": test_mae_loss, "test_RMSE": test_rmse})
        
        # save the model
        if accelerator is not None:
            accelerator.save_model(model, path)
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')

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
    if args.lradj == 'type1':
        # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch > args.least_epochs:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - args.least_epochs) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
            else:
                print(f'{args.lradj}| Updating learning rate to {lr}')


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

def vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion):
    total_preds, total_references = [], []
    model.eval()
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights) in tqdm(enumerate(vali_loader)):
            cycle_curve_data = cycle_curve_data.float()# [B, S, N]
            curve_attn_mask = curve_attn_mask.float()
            labels = labels.float()

            # encoder - decoder
            outputs = model(cycle_curve_data, curve_attn_mask)
            # self.accelerator.wait_for_everyone()
            std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            
    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)
    model.train()
    return rmse, mae, mape


def vali_batteryLifeLLM_relative_absolute_alignment(args, accelerator, model, vali_data, vali_loader, criterion):
    model.eval()
    euclidean_dist = nn.PairwiseDistance(p=2)
    total_align2_loss = []
    total_align1_loss = []
    with torch.no_grad():
        for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights, class_centers, end_input_ids, end_attn_mask, label_prompt_embedding, label_input_ids, label_attention_mask) in enumerate(vali_loader):
            life_class = life_class
            scaled_life_class = scaled_life_class.long()
            cycle_curve_data = cycle_curve_data.float()
            curve_attn_mask = curve_attn_mask.float() # [B, L]
            labels = labels.float()
            input_ids = input_ids.int()
            attention_mask = attention_mask.int()
            weights = weights.float()
            end_input_ids = end_input_ids.int()
            end_attn_mask = end_attn_mask.int()
            label_input_ids = label_input_ids.int()
            label_attention_mask = label_attention_mask.int()
            label_prompt_embedding = label_prompt_embedding.float()
            cut_off = labels.shape[0]


            _, preds_life_class, llm_out, feature_llm_out, outputs, label_feature_llm_out, label_llm_out = model(cycle_curve_data, curve_attn_mask, 
            input_ids=input_ids, attention_mask=attention_mask, 
            end_input_ids=end_input_ids, end_attn_mask=end_attn_mask, 
            label_input_ids=label_input_ids, label_attention_mask=label_attention_mask)
            
            ''' New alignment method '''
            # relative distance alignment
            N, D = label_llm_out.shape[0], label_llm_out.shape[-1]
            tmp_label_llm_out = label_llm_out.unsqueeze(1) # [N, 1, d_llm]
            tmp_label_llm_out = tmp_label_llm_out.expand(-1, N, -1)
            relative_truth = torch.norm(tmp_label_llm_out - tmp_label_llm_out.transpose(0,1), p=2, dim=-1) / np.sqrt(D) # [N, N]
            relative_truth_std, relative_truth_mean = torch.std(relative_truth), torch.mean(relative_truth)
            relative_truth = (relative_truth - relative_truth_mean) / max(relative_truth_std, 1e-6)
            
            tmp_llm_out = llm_out.unsqueeze(1)  # [N, 1, d_llm]
            tmp_llm_out = tmp_llm_out.expand(-1, N, -1)
            relative_preds = torch.norm(tmp_llm_out - tmp_llm_out.transpose(0,1), p=2, dim=-1) / np.sqrt(D) # [N, N]
            relative_preds = (relative_preds - relative_truth_mean) / max(relative_truth_std, 1e-6)

            mismatch = relative_truth-relative_preds
            mismatch = mismatch.reshape(-1) # [N*N]

            relative_alignment_loss = torch.norm(mismatch, p=2) / (N*N-N)

            # absolute distance alignment
            abs_alignment_loss = torch.norm(label_llm_out - llm_out, p=2, dim=1) / np.sqrt(D)  # [N]
            abs_alignment_loss = torch.mean(abs_alignment_loss)

            align_loss, align_loss2 = accelerator.gather_for_metrics((relative_alignment_loss, abs_alignment_loss))
            total_align1_loss = total_align1_loss + align_loss.detach().cpu().numpy().reshape(-1).tolist()
            total_align2_loss = total_align2_loss + align_loss2.detach().cpu().numpy().reshape(-1).tolist()

    total_align2_loss = np.mean(total_align2_loss)
    total_align1_loss = np.mean(total_align1_loss)

    model.train()
    return total_align1_loss, total_align2_loss

def vali_batteryLifeLLM_alignment(args, accelerator, model, vali_data, vali_loader, criterion):
    model.eval()
    euclidean_dist = nn.PairwiseDistance(p=2)
    total_align2_loss = []
    total_align1_loss = []
    with torch.no_grad():
        for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights, class_centers, end_input_ids, end_attn_mask, label_prompt_embedding) in enumerate(vali_loader):
            life_class = life_class
            scaled_life_class = scaled_life_class.long()
            cycle_curve_data = cycle_curve_data.float()
            curve_attn_mask = curve_attn_mask.float() # [B, L]
            labels = labels.float()
            input_ids = input_ids.int()
            attention_mask = attention_mask.int()
            weights = weights.float()
            end_input_ids = end_input_ids.int()
            end_attn_mask = end_attn_mask.int()
            cut_off = labels.shape[0]

            # encoder - decoder
            _, _, _, _, llm_out = model(cycle_curve_data, curve_attn_mask, 
            input_ids=input_ids, attention_mask=attention_mask, end_input_ids=end_input_ids, end_attn_mask=end_attn_mask)
            # self.accelerator.wait_for_everyone()
            tmp_llm_out = llm_out[:cut_off]


            # minimize the cosine similarit between the embeddings of the learned prompt and label prompt
            cosine_sim = torch.cosine_similarity(tmp_llm_out, label_prompt_embedding, dim=1)
            align_loss = - cosine_sim.float()

            distances = euclidean_dist(tmp_llm_out, label_prompt_embedding) / np.sqrt(label_prompt_embedding.shape[-1])
            # distances = torch.norm(tmp_llm_out-label_llm_out, p=2, dim=1) # [N]
            align_loss2 = distances.float()
            align_loss, align_loss2 = accelerator.gather_for_metrics((align_loss, align_loss2))
            total_align1_loss = total_align1_loss + align_loss.detach().cpu().numpy().reshape(-1).tolist()
            total_align2_loss = total_align2_loss + align_loss2.detach().cpu().numpy().reshape(-1).tolist()

    total_align2_loss = np.mean(total_align2_loss)
    total_align1_loss = np.mean(total_align1_loss)

    model.train()
    return total_align1_loss, total_align2_loss

def vali_labelHead(args, accelerator, model, vali_data, vali_loader, criterion):
    model.eval()
    total_preds, total_references = [], []
    std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
    with torch.no_grad():
        for i, (_, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, _, end_input_ids, end_attn_mask, label_prompt_embedding, label_input_ids, label_attention_mask, _) in enumerate(vali_loader):
            cycle_curve_data = cycle_curve_data.float()# [B, S, N]
            curve_attn_mask = curve_attn_mask.float()
            labels = labels.float()
            input_ids = input_ids.int()
            attention_mask = attention_mask.int()
            end_input_ids = end_input_ids.int()
            end_attn_mask = end_attn_mask.int()
            label_input_ids = label_input_ids.int()
            label_attention_mask = label_attention_mask.int()
            # encoder - decoder
            _, _, _, _, outputs, _, _ = model(cycle_curve_data, curve_attn_mask, input_ids=input_ids, attention_mask=attention_mask, 
                                           end_input_ids=end_input_ids, end_attn_mask=end_attn_mask, 
                                           label_input_ids=label_input_ids, label_attention_mask=label_attention_mask)
            # self.accelerator.wait_for_everyone()
            
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            
    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)
    model.train()
    return rmse, mae, mape

def vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False):
    model.eval()
    total_preds, total_references = [], []
    total_seen_unseen_ids = []
    std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, labels, _,  _, DKP_embeddings, seen_unseen_ids, cathode_masks) in enumerate(vali_loader):
            cycle_curve_data = cycle_curve_data.float()# [B, S, N]
            curve_attn_mask = curve_attn_mask.float()
            labels = labels.float()
            cathode_masks = cathode_masks.float()

            # encoder - decoder
            outputs, _, _, _, _, _, _, _ = model(cycle_curve_data, curve_attn_mask, DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks)
            # self.accelerator.wait_for_everyone()
            
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
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
        seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        if len(unseen_preds) > 0:
            unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
        else:
            unseen_mape = -10000

        # alpha-acc1 
        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=args.alpha1)
        seen_alpha_acc1 = hit_num / len(seen_references) * 100

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha1)
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc1 = -10000

        # alpha-acc2
        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=args.alpha2)
        seen_alpha_acc2 = hit_num / len(seen_references) * 100

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

def vali(args, accelerator, model, vali_data, vali_loader, criterion):
    mae_metric = evaluate.load('./utils/mae')
    mse_metric = evaluate.load('./utils/mse')
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_y_masks, batch_used_cycles, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float()
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float()
            batch_y_masks = batch_y_masks.float()
            batch_y_mark = batch_y_mark.float()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # self.accelerator.wait_for_everyone()
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            masks = batch_y_masks.long()
            
            outputs, batch_y, masks = accelerator.gather_for_metrics((outputs, batch_y, masks))
            masks = masks.reshape(*masks.shape, 1)

            outputs, batch_y = outputs[masks==1], batch_y[masks==1] # get the valid predictions and labels
            std, mean_value = np.sqrt(vali_data.scaler.var_[-1]), vali_data.scaler.mean_[-1]
            transformed_preds = outputs * std + mean_value
            transformed_labels = batch_y * std + mean_value
            
            mae_metric.add_batch(
                predictions = transformed_preds,
                references = transformed_labels
            )
            mse_metric.add_batch(
                predictions = transformed_preds,
                references = transformed_labels
            )
            
    mae_res = mae_metric.compute()
    mse_res = mse_metric.compute()
    model.train()
    return mse_res['mse'], mae_res['mae']


# def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
#     total_loss = []
#     total_mae_loss = []
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
#             batch_x = batch_x.float()
#             batch_y = batch_y.float()

#             batch_x_mark = batch_x_mark.float()
#             batch_y_mark = batch_y_mark.float()

#             # decoder input
#             dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
#                 accelerator.device)
#             # encoder - decoder
#             if args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     if args.output_attention:
#                         outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             else:
#                 if args.output_attention:
#                     outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                 else:
#                     outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             # self.accelerator.wait_for_everyone()
#             f_dim = -1 if args.features == 'MS' else 0
#             outputs = outputs[:, -args.pred_len:, :]
#             batch_y = batch_y[:, -args.pred_len:, :]

#             pred = outputs.detach()
#             true = batch_y.detach()

#             loss = criterion(pred, true)
#             masks = batch_x_mark
#             loss = torch.sum(loss[masks==1]) / torch.sum(masks)

#             transformed_pred = pred.cpu().numpy().reshape(-1,pred.shape[-1])
#             transformed_true = true.cpu().numpy().reshape(-1,true.shape[-1])
#             transformed_pred = vali_data.inverse_transform(transformed_pred)
#             transformed_true = vali_data.inverse_transform(transformed_true)

#             transformed_pred = transformed_pred[:,-1:]
#             transformed_true = transformed_true[:,-1:]
#             masks = masks.detach().cpu().numpy().reshape(-1,1)
#             mae_loss = np.abs(transformed_pred - transformed_true)
#             mae_loss = mae_loss[masks==1]

#             total_loss.append(loss.item())
#             total_mae_loss.append(mae_loss)

#     total_loss = np.average(total_loss)
#     total_mae_loss = np.concatenate(total_mae_loss, axis=0)
#     total_mae_loss = np.average(total_mae_loss)

#     model.train()
#     return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float()
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float()
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y))
        batch_y_mark = torch.ones(true.shape)
        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content