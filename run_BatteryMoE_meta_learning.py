import os
import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from tqdm import tqdm
import learn2learn as l2l
from utils.tools import train_model_course, get_parameter_number, is_training_label_model, split_meta_domains
from utils.losses import bmc_loss, Battery_life_alignment_CL_loss, DG_loss, domain_averaged_MSELoss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryMoE_Hyper, BatteryMoE_Hyper_DKP, baseline_CPTransformerMoE, BatteryMoE_PCA_Transformer, baseline_CPMLPMoE
import pickle
import wandb
from data_provider.data_factory import data_provider_LLMv2
import time
import random
import torch.nn.functional as F
import numpy as np
import datetime
import copy
import joblib
from data_provider.gate_masker import gate_masker
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import json
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "true"
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content

parser = argparse.ArgumentParser(description='BatteryLifeLLM')

def set_seed(seed):
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
parser.add_argument('--low_d_ff', type=int, default=32, help='dimension of low rank in the matrix')
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
parser.add_argument('--num_domains', type=int, default=2, help='the minimum number of domains required in a batch of training samples')
parser.add_argument('--meta_test_percentage', type=int, default=25, help='the percentage of the meta-test domains in each iteration')
parser.add_argument('--meta_test_loss_weight', type=float, default=1.0, help='the weigth of the meta-test loss')
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--T0', type=int, default=2, help='T0 for CosineAnnealingWarmRestarts')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--meta_learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--lradj_factor', type=float, default=0.5, help='the learning rate decay factor')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--warm_up_epoches', type=int, default=0, help='The epoch number for linear Warmup')

# MoE definition
parser.add_argument('--num_condition_experts', type=int, default=2, help="The very specialized experts for one aging condition")
parser.add_argument('--num_hyper_experts', type=int, default=2, help="The number of the hyper experts")
parser.add_argument('--num_views', type=int, default=4, help="The number of the views")
parser.add_argument('--num_general_experts', type=int, default=2, help="The number of the general experts")
parser.add_argument('--num_experts', type=int, default=6, help="The number of the expert")
parser.add_argument('--cathode_experts', type=int, default=13, help="The number of the expert models for proecessing different cathodes")
parser.add_argument('--temperature_experts', type=int, default=20, help="The number of the expert models for proecessing different temperatures")
parser.add_argument('--format_experts', type=int, default=21, help="The number of the expert models for proecessing different formats")
parser.add_argument('--anode_experts', type=int, default=11, help="The number of the expert models for proecessing different anodes")
parser.add_argument('--noisy_gating', action='store_true', default=False, help='Set True to use Noisy Gating')
parser.add_argument('--topK', type=int, default=2, help='The number of the experts used to do the prediction')
parser.add_argument('--importance_weight', type=float, default=0.0, help='The loss weight for balancing expert utilization')
parser.add_argument('--use_ReMoE', action='store_true', default=False, help='Set True to use relu router')

parser.add_argument('--use_guide', action='store_true', default=False, help='Set True to use guidance loss to guide the gate to capture the assigned gating.')
parser.add_argument('--gamma', type=float, default=1.0, help='The loss weight for domain-knowledge guidance')
parser.add_argument('--use_LB', action='store_true', default=False, help='Set True to use Load Balancing loss')
parser.add_argument('--use_PCA', action='store_true', default=False, help='Set True to use prompt embeddings processed by PCA')

# Pretrain
parser.add_argument('--Pretrained_model_path', type=str, default='', help='The path to the saved pretrained model parameters')

# Ablation Study
parser.add_argument('--wo_DKPrompt', action='store_true', default=False, help='Set True to remove domain knowledge prompt')

# BatteryFormer
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')

# Evaluation alpha-accuracy
parser.add_argument('--alpha1', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')

args = parser.parse_args()

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
print(args.__dict__)

if args.use_PCA:
    # Automatically find d_llm dimension
    if args.seed == 2021:
        tmp = pickle.load(open(f'{args.root_path}/training_DKP_embed_all_pca.pkl', 'rb'))
    elif args.seed == 2024:
        tmp = pickle.load(open(f'{args.root_path}/training_DKP_embed_all2024_pca.pkl', 'rb'))
    elif args.seed == 42:
        tmp = pickle.load(open(f'{args.root_path}/training_DKP_embed_all42_pca.pkl', 'rb'))
    else:
        raise Exception('add the prompt emebeddings for the seed here')

    args.d_llm = list(tmp.values())[0].shape[1]
    args.__dict__['d_llm'] = list(tmp.values())[0].shape[1]

if args.Pretrained_model_path:
    pretrained = True
else:
    pretrained = False
    
for ii in range(args.itr):
    # setting record of experiments
    # setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_woDKPr{}_pretrained{}_tl{}'.format(
    #     args.model,
    #     args.seq_len,
    #     args.learning_rate,
    #     args.d_model,
    #     args.n_heads,
    #     args.e_layers,
    #     args.d_layers,
    #     args.d_ff,
    #     args.llm_layers, args.use_LoRA, args.lradj, args.dataset, args.use_guide, args.use_LB, args.loss, args.wd, args.weighted_loss, args.wo_DKPrompt, pretrained, args.tune_layers)
    setting = '{}_sl{}_lr{}_mlr{}_dm{}_nh{}_el{}_dl{}_df{}_dfg{}_lradj{}_dataset{}_guide{}_LB{}_loss{}_wd{}_wl{}_dr{}_bf{}_NumE{}_NumGE{}_NumHE{}_NumCE{}_K{}_PCA{}_NDomain{}_MTestW{}_seed{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.meta_learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.low_d_ff,
        args.lradj, args.dataset, args.use_guide, args.use_LB, args.loss, args.wd, args.weighted_loss, args.dropout, 
        args.bottleneck_factor, args.num_experts, args.num_general_experts, args.num_hyper_experts, args.num_condition_experts, args.topK, args.use_PCA, args.num_domains, args.meta_test_loss_weight, args.seed)

    data_provider_func = data_provider_LLMv2
    if args.model == 'baseline_CPTransformerMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPTransformerMoE.Model(model_config)
    elif args.model == 'BatteryMoE_Hyper':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_Hyper.Model(model_config)
    elif args.model == 'BatteryMoE_Hyper_DKP':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_Hyper_DKP.Model(model_config)
    elif args.model == 'BatteryMoE_PCA_Transformer':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_PCA_Transformer.Model(model_config)
    elif args.model == 'baseline_CPMLPMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPMLPMoE.Model(model_config)
    else:
        raise Exception('Not Implemented')

    model = model.bfloat16().cuda()
    tokenizer = None
    if model.tokenizer:
        tokenizer = model.tokenizer
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    if not 'MIX_all' in args.dataset:
        temperature2mask = gate_masker.MIX_large_temperature2mask
        format2mask = gate_masker.MIX_large_format2mask
        cathodes2mask = gate_masker.MIX_large_cathodes2mask
        anode2mask = gate_masker.MIX_large_anode2mask
    else:
        temperature2mask = gate_masker.MIX_all_temperature2mask
        format2mask = gate_masker.MIX_all_format2mask
        cathodes2mask = gate_masker.MIX_all_cathodes2mask
        anode2mask = gate_masker.MIX_all_anodes2mask

    train_data, train_loader = data_provider_func(args, 'train', tokenizer, temperature2mask=temperature2mask, 
                                                  format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask, meta_learning=True)
    label_scaler = train_data.return_label_scaler()        
    
    print("Loading training samples......")
    print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask)
    print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask)
    
    if os.path.exists(path):
        del_files(path)  # delete checkpoint files
        print(f'success delete {path}')
    
    os.makedirs(path, exist_ok=True)
    joblib.dump(label_scaler, f'{path}/label_scaler')

    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    wandb.init(
    # set the wandb project where this run will be logged
    project="BatteryMoE_metaW",
    
    # track hyperparameters and run metadata
    config=args.__dict__,
    name=nowtime
    )



    para_res = get_parameter_number(model)
    print(para_res)

    # Print layer names and parameter counts
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Number of parameters: {param.numel()}")

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(args, accelerator=None, patience=args.patience, least_epochs=args.least_epochs)

    # trained_parameters = []
    # trained_parameters_names = []
    # for name, p in model.named_parameters():
    #     if p.requires_grad is True:
    #         trained_parameters_names.append(name)
    #         trained_parameters.append(p)

    # print(f'Trainable parameters are: {trained_parameters_names}')
    maml = l2l.algorithms.MAML(model, lr=args.meta_learning_rate, first_order=False, allow_unused=True)
    if args.wd == 0:
        model_optim = optim.Adam(maml.parameters(), weight_decay=args.wd, lr=args.learning_rate)
    else:
        model_optim = optim.AdamW(maml.parameters(), weight_decay=args.wd, lr=args.learning_rate)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optim, T_0=args.T0, eta_min=0, T_mult=2, last_epoch=-1)
    # criterion = nn.MSELoss(reduction='none') 
    criterion = domain_averaged_MSELoss()

    prompt_adapter_loss = nn.CrossEntropyLoss()
    euclidean_dist = nn.PairwiseDistance(p=2)


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
        iter_count = 0
        total_loss = 0
        total_guidance_loss = 0
        total_alignment_loss = 0
        total_LB_loss = 0
        total_align2_loss = 0
        total_label_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_guidance_loss = 0
        print_alignment_loss = 0
        print_LB_loss = 0
        print_align2_loss = 0
        print_label_loss = 0


        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []
        for i, (cycle_curve_data, curve_attn_mask, labels, weights, _, DKP_embeddings, _, cathode_masks, temperature_masks, format_masks, anode_masks, combined_masks, domain_ids) in enumerate(train_loader):
            if epoch < args.warm_up_epoches:
                # adjust the learning rate
                warm_up_lr = args.learning_rate * (len(train_loader)*epoch + i + 1) / (args.warm_up_epoches*len(train_loader))
                for param_group in model_optim.param_groups:
                    param_group['lr'] = warm_up_lr

                if (i + 1) % 5 == 0:
                        print(f'Warmup | Updating learning rate to {warm_up_lr}')

            learner = maml.clone()
            iter_count += 1

            cycle_curve_data = cycle_curve_data.to(torch.bfloat16).cuda()
            curve_attn_mask = curve_attn_mask.to(torch.bfloat16).cuda()
            DKP_embeddings = DKP_embeddings.to(torch.bfloat16).cuda()
            cathode_masks = cathode_masks.to(torch.bfloat16).cuda()
            temperature_masks = temperature_masks.to(torch.bfloat16).cuda()
            format_masks = format_masks.to(torch.bfloat16).cuda()
            anode_masks = anode_masks.to(torch.bfloat16).cuda()
            combined_masks = combined_masks.to(torch.bfloat16).cuda()
            labels = labels.cuda()
            weights = weights.cuda()
            domain_ids = domain_ids.int().cuda()

            meta_train_indices, meta_test_indices = split_meta_domains(domain_ids, args.meta_test_percentage)
            # prepare the meta-train and meta-test data
            # meta-train data
            meta_train_cycle_curve_data = cycle_curve_data[meta_train_indices]
            meta_train_curve_attn_mask = curve_attn_mask[meta_train_indices]
            meta_train_DKP_embeddings = DKP_embeddings[meta_train_indices]
            meta_train_cathode_masks = cathode_masks[meta_train_indices]
            meta_train_temperature_masks = temperature_masks[meta_train_indices]
            meta_train_format_masks = format_masks[meta_train_indices]
            meta_train_anode_masks = anode_masks[meta_train_indices]
            meta_train_combined_masks = combined_masks[meta_train_indices]
            meta_train_labels = labels[meta_train_indices]
            meta_train_weights = weights[meta_train_indices]
            meta_train_domain_ids = domain_ids[meta_train_indices]

            # meta-test data
            meta_test_cycle_curve_data = cycle_curve_data[meta_test_indices]
            meta_test_curve_attn_mask = curve_attn_mask[meta_test_indices]
            meta_test_DKP_embeddings = DKP_embeddings[meta_test_indices]
            meta_test_cathode_masks = cathode_masks[meta_test_indices]
            meta_test_temperature_masks = temperature_masks[meta_test_indices]
            meta_test_format_masks = format_masks[meta_test_indices]
            meta_test_anode_masks = anode_masks[meta_test_indices]
            meta_test_combined_masks = combined_masks[meta_test_indices]
            meta_test_labels = labels[meta_test_indices]
            meta_test_weights = weights[meta_test_indices]
            meta_test_domain_ids = domain_ids[meta_test_indices]

            # encoder - decoder
            outputs, prompt_scores, llm_out, feature_llm_out, _, alpha_exponent, aug_loss, guide_loss = learner(meta_train_cycle_curve_data, meta_train_curve_attn_mask, 
            DKP_embeddings=meta_train_DKP_embeddings, cathode_masks=meta_train_cathode_masks, temperature_masks=meta_train_temperature_masks, format_masks=meta_train_format_masks, 
            anode_masks=meta_train_anode_masks, combined_masks=meta_train_combined_masks)


            loss = criterion(outputs.reshape(-1), meta_train_labels.reshape(-1), meta_train_domain_ids)
            # loss = criterion(outputs, meta_train_labels)
            # loss = torch.mean(loss * meta_train_weights)
            final_loss = loss
            if args.num_experts > 1 and args.use_LB:
                # load balancing loss
                importance_loss = args.importance_weight * aug_loss.float() * args.num_experts
                print_LB_loss = importance_loss.detach().float()
                final_loss = final_loss + importance_loss

            if args.use_guide:
                # guidance loss
                guide_loss = args.gamma * guide_loss
                print_guidance_loss = guide_loss.detach().float()
                final_loss = final_loss + guide_loss

            # collect the prediction on the meta-train domains
            transformed_preds = outputs * std + mean_value
            transformed_labels = meta_train_labels  * std + mean_value
            all_predictions, all_targets = transformed_preds, transformed_labels
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            learner.adapt(final_loss) # adapt the model on the meta-train domains
            # evaluate the model on the meta-test domains
            outputs, prompt_scores, llm_out, feature_llm_out, _, alpha_exponent, aug_loss, guide_loss = learner(meta_test_cycle_curve_data, meta_test_curve_attn_mask, 
            DKP_embeddings=meta_test_DKP_embeddings, cathode_masks=meta_test_cathode_masks, temperature_masks=meta_test_temperature_masks, format_masks=meta_test_format_masks, 
            anode_masks=meta_test_anode_masks, combined_masks=meta_test_combined_masks)

            loss = criterion(outputs.reshape(-1), meta_test_labels.reshape(-1), meta_test_domain_ids)
            # loss = criterion(outputs, meta_test_labels)
            # loss = torch.mean(loss * meta_test_weights)
            
            meta_test_final_loss = loss
            if args.num_experts > 1 and args.use_LB:
                # load balancing loss
                importance_loss = args.importance_weight * aug_loss.float() * args.num_experts
                print_LB_loss = importance_loss.detach().float()
                meta_test_final_loss = meta_test_final_loss + importance_loss

            if args.use_guide:
                # guidance loss
                guide_loss = args.gamma * guide_loss
                print_guidance_loss = guide_loss.detach().float()
                meta_test_final_loss = meta_test_final_loss + guide_loss

            final_loss = final_loss + args.meta_test_loss_weight * meta_test_final_loss

            print_label_loss = loss.item()
            print_loss = final_loss.item()
            
            total_loss += final_loss.item()
            total_guidance_loss += print_guidance_loss
            total_LB_loss += print_LB_loss
            total_label_loss += print_label_loss

            transformed_preds = outputs * std + mean_value
            transformed_labels = meta_test_labels  * std + mean_value
            all_predictions, all_targets = transformed_preds, transformed_labels
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()

            model_optim.zero_grad()
            final_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # gradient clipping
            model_optim.step()
            


            if (i + 1) % 5 == 0:
                print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {print_label_loss:.7f} | guidance_loss: {print_guidance_loss:.7f} | LB loss: {print_LB_loss:.7f}')
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                

        train_rmse = root_mean_squared_error(total_references, total_preds)
        train_mape = mean_absolute_percentage_error(total_references, total_preds)
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_batteryLifeLLM(args, None, learner.module, vali_data, vali_loader, criterion)
        test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_batteryLifeLLM(args, None, learner.module, test_data, test_loader, criterion, compute_seen_unseen=True)
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
        total_guidance_loss = total_guidance_loss / len(train_loader)
        total_LB_loss = total_LB_loss / len(train_loader)
        total_label_loss = total_label_loss / len(train_loader)
        print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train label loss: {total_label_loss:.5f} | Train guidance loss: {total_guidance_loss:.5f}| Train LB loss: {total_LB_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali R{args.loss}: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test acc1: {test_alpha_acc1:.4f} | Test MAPE: {test_mape:.7f}")

        wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_RMSE": vali_rmse, "vali_MAPE": vali_mape, "vali_acc1": vali_alpha_acc1, "vali_acc2": vali_alpha_acc2, 
                "test_RMSE": test_rmse, "test_MAPE": test_mape, "test_acc1": test_alpha_acc1, "test_acc2": test_alpha_acc2})
        
        early_stopping(epoch+1, vali_loss, vali_mae_loss, test_mae_loss, learner.module, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if args.lradj != 'CosineAnnealingLR':
            adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            if epoch >= args.warm_up_epoches:
                scheduler.step()
                print('CosineAnnealingWarmRestarts| Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Test 15%-accuracy: {best_test_alpha_acc1:.4f} | Test 10%-accuracy: {best_test_alpha_acc2:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f} | Val 15%-accuracy: {best_vali_alpha_acc1:.4f} | Val 10%-accuracy: {best_vali_alpha_acc2:.4f} ')
print(f'Best model performance: Test Seen MAPE: {best_seen_test_MAPE:.4f} | Test Unseen MAPE: {best_unseen_test_MAPE:.4f}')
print(f'Best model performance: Test Seen 15%-accuracy: {best_seen_test_alpha_acc1:.4f} | Test Unseen 15%-accuracy: {best_unseen_test_alpha_acc1:.4f}')
print(f'Best model performance: Test Seen 10%-accuracy: {best_seen_test_alpha_acc2:.4f} | Test Unseen 10%-accuracy: {best_unseen_test_alpha_acc2:.4f}')
print(path)

wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_RMSE": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "vali_acc1": best_vali_alpha_acc1, "vali_acc2": best_vali_alpha_acc2, 
        "test_RMSE": best_test_RMSE, "test_MAPE":best_test_MAPE, "test_acc1": best_test_alpha_acc1, "test_acc2": best_test_alpha_acc2})
wandb.finish()
