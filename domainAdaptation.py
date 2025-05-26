# curriculum learning
import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import evaluate
from utils.tools import get_parameter_number
from models import BatteryMoE_PESum, baseline_CPTransformerMoE, baseline_CPMLPMoE, BatteryMoE_horizontal_MHv2, BatteryMoE_Sparse
import wandb
from transformers import AutoConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_baseline_DA
import time
import json
import random
import numpy as np
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content
from data_provider.gate_masker import gate_masker
import os
import json
import datetime
from utils.losses import MMD_loss
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_baseline, load_content
parser = argparse.ArgumentParser(description='BatteryLife')

def set_seed(seed):
    accelerate.utils.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--LLM_path', type=str, required=False, default='/home/trf/LLMs/llama2-hf-7b',
                    help='The path to the saved LLM checkpoints')
parser.add_argument('--center_path', type=str, required=False, default='./Centenr_vectors',
                    help='The path to the preset cluster centers')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--dataset', type=str, default='HUST', help='dataset description')
parser.add_argument('--data', type=str, required=False, default='BatteryLifeLLM', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/HUST_dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--num_process', type=int, default=4, help='the number of used GPUs')
# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='what is early life')
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--d_llm', type=int, default=4096, help='the features of llm')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--noDKP_layers', type=int, default=1, help='the number of no DKP layers in the inter-cycle encoder')
parser.add_argument('--bottleneck_factor', type=int, default=2, help='the scale down factor of the bottleneck layer')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--d_ff_gate', type=int, default=32, help='dimension of fcn for gate network')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=10, help='patch length')
parser.add_argument('--stride', type=int, default=10, help='stride')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')
parser.add_argument('--class_num', type=int, default=8, help='The number of life classes')

# optimization
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--T0', type=int, default=2, help='T0 for CosineAnnealingWarmRestarts')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--lradj_factor', type=float, default=0.5, help='the learning rate decay factor')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--top_p', type=float, default=0.5, help='The threshold used to control the number of activated experts')
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--warm_up_epoches', type=int, default=0, help='The epoch number for linear Warmup')

# MoE definition
parser.add_argument('--num_views', type=int, default=4, help="The number of the views")
parser.add_argument('--num_general_experts', type=int, default=2, help="The number of the expert models used to process the battery data when the input itself is used for gating")
parser.add_argument('--num_experts', type=int, default=6, help="The number of the expert models used to process the battery data in encoder")
parser.add_argument('--cathode_experts', type=int, default=13, help="The number of the expert models for proecessing different cathodes")
parser.add_argument('--temperature_experts', type=int, default=20, help="The number of the expert models for proecessing different temperatures")
parser.add_argument('--format_experts', type=int, default=21, help="The number of the expert models for proecessing different formats")
parser.add_argument('--anode_experts', type=int, default=11, help="The number of the expert models for proecessing different anodes")
parser.add_argument('--noisy_gating', action='store_true', default=False, help='Set True to use Noisy Gating')
parser.add_argument('--topK', type=int, default=2, help='The number of the experts used to do the prediction')
parser.add_argument('--importance_weight', type=float, default=0.0, help='The loss weight for balancing expert utilization')
parser.add_argument('--use_ReMoE', action='store_true', default=False, help='Set True to use relu router')

# Contrastive learning
parser.add_argument('--use_guide', action='store_true', default=False, help='Set True to use guidance loss to guide the gate to capture the assigned gating.')
parser.add_argument('--gamma', type=float, default=1.0, help='The loss weight for domain-knowledge guidance')
# Domain generalization
parser.add_argument('--use_LB', action='store_true', default=False, help='Set True to use Load Balancing loss')

# Pretrain
parser.add_argument('--Pretrained_model_path', type=str, default='', help='The path to the saved pretrained model parameters')

# Ablation Study
parser.add_argument('--wo_DKPrompt', action='store_true', default=False, help='Set True to remove domain knowledge prompt')

# BatteryFormer
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')

# Evaluation alpha-accuracy
parser.add_argument('--alpha1', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')

# DA
parser.add_argument('--target_dataset', type=str, default='None', help='the target domain dataset')
parser.add_argument('--DA_weight', type=float, default=1, help='The weight of MMD loss')
parser.add_argument('--DA_target_mseloss_weight', type=float, default=1, help='The weight of target mse loss')

args = parser.parse_args()


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_dfg{}_llmLayers{}_lradj{}_dataset{}_guide{}_LB{}_loss{}_wd{}_wl{}_noDKPL{}_dr{}_bf{}_NumE{}_K{}_seed{}_DA'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.d_ff_gate,
        args.llm_layers, args.lradj, args.dataset, args.use_guide, args.use_LB, args.loss, args.wd, args.weighted_loss, args.noDKP_layers, args.dropout, args.bottleneck_factor, args.num_experts, args.topK, args.seed)


    data_provider_func = data_provider_baseline_DA
    if args.model == 'baseline_CPTransformerMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPTransformerMoE.Model(model_config)
    elif args.model == 'baseline_CPMLPMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPMLPMoE.Model(model_config)
    elif args.model == 'BatteryMoE_PESum':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_PESum.Model(model_config)
    elif args.model == 'BatteryMoE_horizontal_MHv2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_horizontal_MHv2.Model(model_config)
    else:
        raise Exception(f'The {args.model} is not an implemented baseline!')
        
    tokenizer = None
    if model.tokenizer:
        tokenizer = model.tokenizer
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    # if not 'MIX_all' in args.dataset:
    #     temperature2mask = gate_masker.MIX_large_temperature2mask
    #     format2mask = gate_masker.MIX_large_format2mask
    #     cathodes2mask = gate_masker.MIX_large_cathodes2mask
    #     anode2mask = gate_masker.MIX_large_anode2mask
    # else:
    #     temperature2mask = gate_masker.MIX_all_temperature2mask
    #     format2mask = gate_masker.MIX_all_format2mask
    #     cathodes2mask = gate_masker.MIX_all_cathodes2mask
    #     anode2mask = gate_masker.MIX_all_anodes2mask
    temperature2mask = gate_masker.MIX_all_temperature2mask
    format2mask = gate_masker.MIX_all_format2mask
    cathodes2mask = gate_masker.MIX_all_cathodes2mask
    anode2mask = gate_masker.MIX_all_anodes2mask
    
    accelerator.print("Loading training samples......")
    train_data, train_loader, train_target_data, train_target_loader = data_provider_func(args, 'train', tokenizer, target_dataset=args.target_dataset, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask)
    label_scaler = train_data.return_label_scaler()  

    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask)
    
    if accelerator.is_local_main_process and os.path.exists(path):
        del_files(path)  # delete checkpoint files
        accelerator.print(f'success delete {path}')

    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{path}/label_scaler')

    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    if accelerator.is_local_main_process:
        wandb.init(
        # set the wandb project where this run will be logged
        project="BatteryMoE_FixPrompt_DA",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )
    # args.content = load_content(args)

    # load LoRA
    # print the module name

    # for n, m in model.named_modules():
    #     accelerator.print(n, m)


    para_res = get_parameter_number(model)
    accelerator.print(para_res)
        
    for name, module in model._modules.items():
        accelerator.print(name," : ",module)
    
    
    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(args, accelerator=accelerator, patience=args.patience, least_epochs=args.least_epochs)

    trained_parameters = []
    trained_parameters_names = []
    for name, p in model.named_parameters():
        if p.requires_grad is True:
            trained_parameters_names.append(name)
            trained_parameters.append(p)

    accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
    if args.wd == 0:
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)
    else:
        model_optim = optim.AdamW(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optim, T_0=args.T0, eta_min=0, T_mult=2, last_epoch=-1)
    criterion = nn.MSELoss(reduction='none') 


    life_class_criterion = nn.MSELoss() 
    MMD_loss = MMD_loss()

    train_loader, train_target_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, train_target_loader, vali_loader, test_loader, model, model_optim, scheduler)
    best_vali_loss = float('inf')
    best_vali_MAE, best_test_MAE = 0, 0
    best_vali_RMSE, best_test_RMSE = 0, 0
    best_vali_alpha_acc1, best_test_alpha_acc1 = 0, 0 
    best_vali_alpha_acc2, best_test_alpha_acc2 = 0, 0 

    best_seen_vali_alpha_acc1, best_seen_test_alpha_acc1 = 0, 0
    best_seen_vali_alpha_acc2, best_seen_test_alpha_acc2 = 0, 0
    best_unseen_vali_alpha_acc1, best_unseen_test_alpha_acc1 = 0, 0
    best_unseen_vali_alpha_acc2, best_unseen_test_alpha_acc2 = 0, 0

    best_vali_MAPE, best_test_MAPE = 0, 0
    best_seen_vali_MAPE, best_seen_test_MAPE = 0, 0
    best_unseen_vali_MAPE, best_unseen_test_MAPE = 0, 0

    for epoch in range(args.train_epochs):
        mae_metric = evaluate.load('./utils/mae')
        mape_metric = evaluate.load('./utils/mape')
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        total_lc_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_cl_loss = 0
        print_life_class_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []

        iterator_source = iter(train_loader)
        iterator_target = iter(train_target_loader)
        for i in range(len(train_loader)):
            with accelerator.accumulate(model):
                cycle_curve_data, curve_attn_mask, labels, weights, _, DKP_embeddings, _, cathode_masks, temperature_masks, format_masks, anode_masks, combined_masks, _ = next(iterator_source)
                target_cycle_curve_data, target_curve_attn_mask, target_labels, target_weights, _, target_DKP_embeddings, _, target_cathode_masks, target_temperature_masks, target_format_masks, target_anode_masks, target_combined_masks, _ = next(iterator_target)
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                if epoch < args.warm_up_epoches:
                    # adjust the learning rate
                    warm_up_lr = args.learning_rate * (len(train_loader)*epoch + i + 1) / (args.warm_up_epoches*len(train_loader))
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = warm_up_lr

                    if (i + 1) % 5 == 0:
                        if accelerator is not None:
                            accelerator.print(f'Warmup | Updating learning rate to {warm_up_lr}')
                        else:
                            print(f'Warmup | Updating learning rate to {warm_up_lr}')

                model_optim.zero_grad()
                iter_count += 1
                
                # cycle_curve_data = cycle_curve_data.float() # [B, L, num_variables, fixed_length_of_curve]
                # curve_attn_mask = curve_attn_mask.float() # [B, L]
                # DKP_embeddings = DKP_embeddings.float()
                # cathode_masks = cathode_masks.float()
                # anode_masks = anode_masks.float()
                # temperature_masks = temperature_masks.float()
                # format_masks = format_masks.float()
                # combined_masks = combined_masks.float()
                # # cluster_labels = cluster_labels.long()
                # labels = labels.float()
                # weights = weights.float()


                # target_cycle_curve_data = target_cycle_curve_data.float() # [B, L, num_variables, fixed_length_of_curve]
                # target_curve_attn_mask = target_curve_attn_mask.float() # [B, L]
                # target_DKP_embeddings = target_DKP_embeddings.float()
                # target_cathode_masks = target_cathode_masks.float()
                # target_anode_masks = target_anode_masks.float()
                # target_temperature_masks = target_temperature_masks.float()
                # target_format_masks = target_format_masks.float()
                # target_combined_masks = target_combined_masks.float()
                # # cluster_labels = cluster_labels.long()
                # target_labels = target_labels.float()
                # target_weights = target_weights.float()
                
                
                # encoder - decoder
                outputs, _, embeddings, _, _, _, _, _ = model(cycle_curve_data, curve_attn_mask, 
                DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks, temperature_masks=temperature_masks, format_masks=format_masks, 
                anode_masks=anode_masks, combined_masks=combined_masks)

                target_outputs, _, target_embeddings, _, _, _, _, _ =  model(target_cycle_curve_data, target_curve_attn_mask, 
                DKP_embeddings=target_DKP_embeddings, cathode_masks=target_cathode_masks, temperature_masks=target_temperature_masks, format_masks=target_format_masks, 
                anode_masks=target_anode_masks, combined_masks=target_combined_masks)

                cut_off = labels.shape[0]
                    
                if args.loss == 'MSE':
                    loss = criterion(outputs[:cut_off], labels)
                    loss = torch.mean(loss * weights)
                    target_loss = criterion(target_outputs[:cut_off], target_labels)
                    target_loss = torch.mean(target_loss * target_weights)

                    mmd_loss = MMD_loss(embeddings, target_embeddings)
                    loss = loss + args.DA_target_mseloss_weight * target_loss + args.DA_weight * mmd_loss

                    
                label_loss = loss.detach().float()
                
                
                print_loss = loss.detach().float()
                
                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss
                total_lc_loss += print_life_class_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
                
                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
                accelerator.backward(loss)
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | lc_loss: {print_life_class_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        train_rmse = root_mean_squared_error(total_references, total_preds)
        train_mape = mean_absolute_percentage_error(total_references, total_preds)
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion)
        test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_batteryLifeLLM(args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True)
        vali_loss = vali_mape

        
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            best_vali_MAE = vali_mae_loss
            best_test_MAE = test_mae_loss
            best_vali_RMSE = vali_rmse
            best_test_RMSE = test_rmse
            best_vali_MAPE = vali_mape
            best_test_MAPE = test_mape

            # alpha-accuracy
            best_vali_alpha_acc1 = vali_alpha_acc1
            best_vali_alpha_acc2 = vali_alpha_acc2
            best_test_alpha_acc1 = test_alpha_acc1
            best_test_alpha_acc2 = test_alpha_acc2

            # seen, unseen
            best_seen_test_MAPE = test_seen_mape
            best_unseen_test_MAPE = test_unseen_mape
            best_seen_test_alpha_acc1 = test_seen_alpha_acc1
            best_unseen_test_alpha_acc1 = test_unseen_alpha_acc1
            best_seen_test_alpha_acc2 = test_seen_alpha_acc2
            best_unseen_test_alpha_acc2 = test_unseen_alpha_acc2
            
        train_loss = total_loss / len(train_loader)
        total_cl_loss = total_cl_loss / len(train_loader)
        total_lc_loss = total_lc_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train lc loss: {total_lc_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        
        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_RMSE": vali_rmse, "vali_MAPE": vali_mape, "vali_acc1": vali_alpha_acc1, "vali_acc2": vali_alpha_acc2, 
                       "test_RMSE": test_rmse, "test_MAPE": test_mape, "test_acc1": test_alpha_acc1, "test_acc2": test_alpha_acc2})
        
        early_stopping(epoch+1, vali_loss, vali_mae_loss, test_mae_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            accelerator.set_trigger()
            
        if accelerator.check_trigger():
            break
        
        if accelerator.is_local_main_process:
            if args.lradj != 'CosineAnnealingLR':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                if epoch >= args.warm_up_epoches:
                    scheduler.step()
                    accelerator.print('CosineAnnealingWarmRestarts| Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Test 15%-accuracy: {best_test_alpha_acc1:.4f} | Test 10%-accuracy: {best_test_alpha_acc2:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f} | Val 15%-accuracy: {best_vali_alpha_acc1:.4f} | Val 10%-accuracy: {best_vali_alpha_acc2:.4f} ')
accelerator.print(f'Best model performance: Test Seen MAPE: {best_seen_test_MAPE:.4f} | Test Unseen MAPE: {best_unseen_test_MAPE:.4f}')
accelerator.print(f'Best model performance: Test Seen 15%-accuracy: {best_seen_test_alpha_acc1:.4f} | Test Unseen 15%-accuracy: {best_unseen_test_alpha_acc1:.4f}')
accelerator.print(f'Best model performance: Test Seen 10%-accuracy: {best_seen_test_alpha_acc2:.4f} | Test Unseen 10%-accuracy: {best_unseen_test_alpha_acc2:.4f}')
accelerator.print(path)
accelerator.set_trigger()
if accelerator.check_trigger() and accelerator.is_local_main_process:
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_RMSE": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "vali_acc1": best_vali_alpha_acc1, "vali_acc2": best_vali_alpha_acc2, 
               "test_RMSE": best_test_RMSE, "test_MAPE":best_test_MAPE, "test_acc1": best_test_alpha_acc1, "test_acc2": best_test_alpha_acc2})
    wandb.finish()