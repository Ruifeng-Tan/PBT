import os
import argparse
import torch
import accelerate
from accelerate import DistributedDataParallelKwargs, load_checkpoint_in_model, Accelerator, DeepSpeedPlugin
from torch import nn, optim
from tqdm import tqdm
from utils.tools import get_parameter_number
from utils.losses import DG_loss, Alignment_loss, AverageRnCLoss, WeightedRnCLoss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryMoE_Hyper_CropAugIMPR2, PBT, baseline_CPTransformerMoE, BatteryMoE_Hyper_CropAugIMP, baseline_CPMLPMoE, CPMLP, CPTransformer_ablation
from layers.Adapters import PBTtLayerWithAdapter, PBTCPLayerWithAdapter
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

def add_adapters_to_PBT(args, model, adapter_size=64):
    # original_layer = model.flattenIntraCycleLayer
    # model.flattenIntraCycleLayer = PBTtLayerWithAdapter(
    #     original_layer, 
    #     adapter_size=adapter_size
    # )

    for i in range(len(model.intra_MoE_layers)):
        # add adapters to intra-cycle encoder layers
        original_layer = model.intra_MoE_layers[i]
        model.intra_MoE_layers[i] = PBTtLayerWithAdapter(
            args,
            original_layer, 
            adapter_size=adapter_size
        )
    
    for i in range(len(model.inter_MoE_layers)):
        # add adapters to inter-cycle encoder layers
        original_layer = model.inter_MoE_layers[i]
        model.inter_MoE_layers[i] = PBTtLayerWithAdapter(
            args,
            original_layer, 
            adapter_size=adapter_size
        )


    return model

def add_adapters_to_PBT_withCP(args, model, adapter_size=64):
    original_layer = model.flattenIntraCycleLayer
    model.flattenIntraCycleLayer = PBTCPLayerWithAdapter(
        args,
        original_layer, 
        adapter_size=adapter_size
    )

    for i in range(len(model.intra_MoE_layers)):
        # add adapters to intra-cycle encoder layers
        original_layer = model.intra_MoE_layers[i]
        model.intra_MoE_layers[i] = PBTtLayerWithAdapter(
            args,
            original_layer, 
            adapter_size=adapter_size
        )
    
    for i in range(len(model.inter_MoE_layers)):
        # add adapters to inter-cycle encoder layers
        original_layer = model.inter_MoE_layers[i]
        model.inter_MoE_layers[i] = PBTtLayerWithAdapter(
            args,
            original_layer, 
            adapter_size=adapter_size
        )


    return model

def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "true"
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM

parser = argparse.ArgumentParser(description='BatteryLifeLLM')

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
parser.add_argument('--num_domains', type=int, default=4, help='the number of domains in a training batch')
parser.add_argument('--use_domainSampler', action='store_true', default=False, help='set True to use domain sampler when loading training samples')
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
parser.add_argument('--adapter_size', type=int, default=16, help='dimension of Adapter for adpater tuning')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--noDKP_layers', type=int, default=1, help='the number of no DKP layers in the inter-cycle encoder')
parser.add_argument('--bottleneck_factor', type=int, default=16, help='the scale down factor of the bottleneck layer')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
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
parser.add_argument('--down_sample_ratio', type=float, default=0.75, help='the down sampling ratio for data augmentation')
parser.add_argument('--weighted_CLDG', action='store_true', default=False, help='use weighted CLDG loss')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature for contrastive learning')
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
parser.add_argument('--g_sigma', type=float, default=0.01, help='the sigma for Gaussian noise')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--top_p', type=float, default=0.5, help='The threshold used to control the number of activated experts')
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--warm_up_epoches', type=int, default=0, help='The epoch number for linear Warmup')
parser.add_argument('--use_guide', action='store_true', default=False, help='Set True to use guidance loss to guide the gate to capture the assigned gating.')
parser.add_argument('--gamma', type=float, default=1.0, help='The loss weight for domain-knowledge guidance')
parser.add_argument('--use_LB', action='store_true', default=False, help='Set True to use Load Balancing loss')
parser.add_argument('--use_aug', action='store_true', help='use data augmentation', default=False)
parser.add_argument('--aug_w', type=float, default=1.0, help='The loss weight for domain-knowledge guidance')

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
parser.add_argument('--ion_experts', type=int, default=6, help="The number of the expert models for proecessing different ion types")
parser.add_argument('--noisy_gating', action='store_true', default=False, help='Set True to use Noisy Gating')
parser.add_argument('--topK', type=int, default=2, help='The number of the experts used to do the prediction')
parser.add_argument('--cycle_topK', type=int, default=2, help='The number of the experts used in CycleMoE layer')
parser.add_argument('--importance_weight', type=float, default=0.0, help='The loss weight for balancing expert utilization')
parser.add_argument('--use_ReMoE', action='store_true', default=False, help='Set True to use relu router')
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

# finetune 
parser.add_argument('--finetune_method', type=str, default='FT', help='the fine-tuning method. [FT, EFT, AT]')
parser.add_argument('--finetune_dataset', type=str, help='the target dataset for model finetuning')
parser.add_argument('--args_path', type=str, help='the path to the pretrained model parameters')

args = parser.parse_args()

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)

# load from the saved path
args_path = args.args_path
dataset = args.finetune_dataset
batch_size = args.batch_size
learning_rate = args.learning_rate
finetune_method = args.finetune_method

args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
args_json['least_epochs'] = args.least_epochs
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args_json['dropout'] = args.dropout
args_json['learning_rate'] = learning_rate
args_json['alpha1'] = args.alpha1
args_json['alpha2'] = args.alpha2
args_json['save_path'] = args.checkpoints
args_json['model'] = args.model
args_json['topK'] = args.topK
args_json['use_aug'] = args.use_aug
args_json['aug_w'] = args.aug_w
args_json['temperature'] = args.temperature
args_json['lradj'] = args.lradj
args_json['patience'] = args.patience
args_json['train_epochs'] = args.train_epochs
args_json['finetune_method'] = args.finetune_method
args_json['adapter_size'] = args.adapter_size
args.__dict__ = args_json

    
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_as{}_le{}_bs{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_mdf{}_lradj{}_{}_guide{}_LB{}_loss{}_wd{}_wl{}_dr{}_gdff{}_E{}_GE{}_K{}_S{}_aug{}_augW{}_tem{}_wDG{}_dsr{}_we{}_ffs{}_{}_{}_seed{}'.format(
        args.model,
        args.dk_factor,
        args.llm_choice,
        args.seq_len,
        args.adapter_size,
        args.least_epochs,
        args.batch_size,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.min_d_ff,
        args.lradj, args.dataset, args.use_guide, args.use_LB, args.loss, args.wd, args.weighted_loss, args.dropout, args.gate_d_ff, 
        args.num_experts, args.num_general_experts,
        args.topK, args.use_domainSampler, args.use_aug, args.aug_w, args.temperature, args.weighted_CLDG, args.down_sample_ratio, args.warm_up_epoches, args.use_dff_scale, trained_dataset, finetune_method, args.seed)


    data_provider_func = data_provider_LLMv2
    if args.model == 'baseline_CPTransformerMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPTransformerMoE.Model(model_config)
    elif args.model == 'PBT':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = PBT.Model(model_config)
    elif args.model == 'BatteryMoE_Hyper_CropAugIMPR2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_Hyper_CropAugIMPR2.Model(model_config)
    elif args.model == 'BatteryMoE_Hyper_CropAugIMP':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryMoE_Hyper_CropAugIMP.Model(model_config)
    elif args.model == 'baseline_CPMLPMoE':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = baseline_CPMLPMoE.Model(model_config)
    elif args.model == 'CPTransformer_ablation':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = CPTransformer_ablation.Model(model_config)
    else:
        raise Exception('Not Implemented')

    
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
        ion2mask = None
    else:
        temperature2mask = gate_masker.MIX_all_temperature2mask
        format2mask = gate_masker.MIX_all_format2mask
        cathodes2mask = gate_masker.MIX_all_cathode2mask
        anode2mask = gate_masker.MIX_all_anode2mask
        ion2mask = gate_masker.MIX_all_ion2mask

    label_scaler = joblib.load(f'{args_path}label_scaler')
    train_data, train_loader = data_provider_func(args, 'train', tokenizer, temperature2mask=temperature2mask, 
                                                  format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask, ion2mask=ion2mask, 
                                                  use_domainSampler=args.use_domainSampler, label_scaler=label_scaler)
    label_scaler = train_data.return_label_scaler()
    
    accelerator.print("Loading training samples......")
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask, ion2mask=ion2mask)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask, ion2mask=ion2mask)
    
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
        project="PBT_FT",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )



    para_res = get_parameter_number(model)
    accelerator.print(para_res)

    # Print layer names and parameter counts
    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(f"Layer: {name} | Number of parameters: {param.numel()}")

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(args, accelerator=accelerator, patience=args.patience, least_epochs=args.least_epochs)

    trained_parameters = []
    trained_parameters_names = []
    use_view_experts = True

    if finetune_method == 'FT':
        # free the general experts and tune other parameters
        for name, p in model.named_parameters():
            if p.requires_grad is True:
                trained_parameters_names.append(name)
                trained_parameters.append(p)
    elif finetune_method == 'EFT':
        # use_view_experts = False
        # tune only the shared experts, normalization and output layer
        for name, p in model.named_parameters():
            if 'general_experts' in name:
                continue
            if p.requires_grad is True:
                trained_parameters_names.append(name)
                trained_parameters.append(p)
    elif finetune_method == 'AT':
        # adapter tuning
        model = add_adapters_to_PBT(args, model, args.adapter_size)
        for name, p in model.named_parameters():
            # only tune the adapters + gate + head + flattenIntraCycleLayer
            if 'adapter' in name or 'gate' in name or 'regression_head' in name or 'flattenIntraCycleLayer' in name:
                if p.requires_grad is True:
                    trained_parameters_names.append(name)
                    trained_parameters.append(p)
    elif finetune_method == 'AT_cp':
        # adapter tuning
        model = add_adapters_to_PBT_withCP(args, model, args.adapter_size) # add adapters before and after that flattenIntra
        for name, p in model.named_parameters():
            # only tune the adapters + gate + head
            if 'adapter' in name or 'gate' in name or 'regression_head' in name:
                if p.requires_grad is True:
                    trained_parameters_names.append(name)
                    trained_parameters.append(p)
    else:
        raise Exception(f'{finetune_method} is not implemented!')

    accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
    if args.wd == 0:
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)
    else:
        model_optim = optim.AdamW(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optim, T_0=args.T0, eta_min=0, T_mult=2, last_epoch=-1)
    criterion = nn.MSELoss(reduction='none') 
    rnc_criterion = WeightedRnCLoss(temperature=args.temperature) if args.weighted_CLDG else AverageRnCLoss(temperature=args.temperature)
    
    load_checkpoint_in_model(model, args_path) # load the pretrained parameters into model
    accelerator.print(f'The model is {args.model}')
    accelerator.print(f'load model from:\n {args_path}')
    accelerator.print(f'Model is loaded!')

    # accelerator.state.select_deepspeed_plugin("BatteryLifeLLM")
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

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
        total_label_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_guidance_loss = 0
        print_alignment_loss = 0
        print_LB_loss = 0
        print_label_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []
        for i, (cycle_curve_data, curve_attn_mask, labels, weights, _, DKP_embeddings, _, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids) in enumerate(train_loader):
            with accelerator.accumulate(model):
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

                # encoder - decoder
                outputs, _, embeddings, _, _, alpha_exponent, aug_loss, guide_loss = model(cycle_curve_data, curve_attn_mask, 
                DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks, temperature_masks=temperature_masks, format_masks=format_masks, 
                anode_masks=anode_masks, combined_masks=combined_masks, ion_type_masks=ion_type_masks, use_aug=args.use_aug, use_view_experts=use_view_experts)
                
                # if args.use_aug:
                #     labels = labels.repeat(int(outputs.shape[0] / labels.shape[0]), 1)
                #     weights = labels.repeat(int(outputs.shape[0] / weights.shape[0]), 1)

                if args.loss == 'MSE':
                    loss = criterion(outputs, labels)
                    loss = torch.mean(loss * weights)
                else:
                    raise Exception('Not implemented!')
                
                final_loss = loss
                if args.num_experts > 1 and args.use_LB:
                    importance_loss = args.importance_weight * aug_loss.float() * args.num_experts
                    print_LB_loss = importance_loss.detach().float()
                    final_loss = final_loss + importance_loss

                if args.use_guide:
                    # contrastive learning
                    guide_loss = args.gamma * guide_loss
                    print_guidance_loss = guide_loss.detach().float()
                    final_loss = final_loss + guide_loss

                if args.use_aug:
                    rnc_loss = args.aug_w * rnc_criterion(embeddings, labels)
                    print_alignment_loss = rnc_loss.detach().float()
                    final_loss = final_loss + rnc_loss


                print_label_loss = loss.item()
                print_loss = final_loss.item()
                
                total_loss += final_loss.item()
                total_guidance_loss += print_guidance_loss
                total_alignment_loss += print_alignment_loss
                total_LB_loss += print_LB_loss
                total_label_loss += print_label_loss

                transformed_preds = outputs * std + mean_value
                transformed_labels = labels * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))

                accelerator.backward(final_loss)
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # gradient clipping
                model_optim.step()
                

                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()


                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {print_label_loss:.7f} | guidance_loss: {print_guidance_loss:.7f} | align_loss: {print_alignment_loss:.7f} | LB loss {print_LB_loss:.7f}')
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
        total_guidance_loss = total_guidance_loss / len(train_loader)
        total_alignment_loss = total_alignment_loss / len(train_loader)
        total_LB_loss = total_LB_loss / len(train_loader)
        total_label_loss = total_label_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train label loss: {total_label_loss:.5f} | Train cl loss: {total_guidance_loss:.5f}| Train align loss: {total_alignment_loss:.5f} | Train LB loss {total_LB_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali R{args.loss}: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test acc1: {test_alpha_acc1:.4f} | Test MAPE: {test_mape:.7f}")
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
