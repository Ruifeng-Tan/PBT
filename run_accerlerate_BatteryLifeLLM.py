# curriculum learning
import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.CMixUp_algorithm import get_batch_kde_mixup_batch, get_batch_kde_mixup_idx, get_similarity
import evaluate

from utils.tools import train_model_course, get_parameter_number, is_training_label_model
from utils.losses import bmc_loss, Battery_life_alignment_CL_loss, DG_loss, Alignment_loss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryLifeLLMv10_LinearFusion_Trial, BatteryLifeLLMv10_TrialP2, BatteryLifeLLMv10_TrialP2_full_SDCM, BatteryLifeLLMv10_TrialP2_noDKP, BatteryLifeLLMv10_TrialP2_noDKP_CM_R, BatteryLifeLLMv10_TrialP2_noDKP_ITCM, BatteryLifeLLMv10_TrialP2_noDKP_ITSDCM, BatteryLifeLLMv10_TrialP2_noDKP_PSDCM, BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask, BatteryLifeLLMv10_TrialP2_noDKP_PTuning, BatteryLifeLLMv10_TrialP2_noDKP_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_P, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_R, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv10_TrialP2_noDKP_selfD, BatteryLifeLLMv11_SDCM, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv13_SDCM_imp, BatteryLifeLLMv18_Stack, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, BatteryLifeLLMv9, TimeLLM, \
            BatteryLifeLLMv7,BatteryLifeLLMv7_LinearHead_aug, BatteryLifeLLMv7_LinearHead2_tuneLLM, BatteryLifeLLMv7_GRUHead_tuneLLM, \
            BatteryLifeLLMv7_MLPHead_tuneLLM, BatteryLifeLLMv7_TransHead, \
            BatteryLifeLLMv7_LinearHead_S, BatteryLifeLLMv8_S_reprogramming, BatteryLifeLLMv9_LLM_noTune_hyper_Trial_P, BatteryLifeLLMv10, \
            BatteryLifeLLMv10_Trial, BatteryLifeLLMv11, BatteryLifeLLMv10_Trial_noCycleP, BatteryLifeLLMv10_Trial_noDKP
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AdaLoraConfig
from data_provider.data_factory import data_provider_LLMv2
import time
import random
import torch.nn.functional as F
import numpy as np
import os
import datetime
import copy
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import json
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
# os.environ["TORCH_USE_CUDA_DSA"] = "true"
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content

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

# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='what is early life')
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--dominant_frequency', type=int, default=100, help='Used in low pass filter')
parser.add_argument('--last_layer', type=int, default=0, help='The layer index for fusion')
parser.add_argument('--d_llm', type=int, default=4096, help='the features of llm')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--lstm_layers', type=int, default=1, help='num of LSTM layers')
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
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')
parser.add_argument('--class_num', type=int, default=8, help='The number of life classes')

# optimization
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--weighted_sampling', action='store_true', default=False, help='use weighted sampling')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--noDG_epochs', type=int, default=-1, help='the train epochs before DG is used')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, 
                    default=1)
parser.add_argument('--mlp', type=int, default=0)

# Contrastive learning
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--beta', type=float, default=1, help='Instance-level alignment weight')
parser.add_argument('--gamma', type=float, default=1, help='Alignment weight')
parser.add_argument('--use_align', action='store_true', default=False, help='Set True to use contrastive learning')

# Deal with long-tail issue
parser.add_argument('--use_hyper', action='store_true', default=False, help='Set True to use HyperNN')
parser.add_argument('--life_class_weight', type=float, default=0.5)

# Domain generalization
parser.add_argument('--use_DG', action='store_true', default=False, help='Set True to use domain generalization')
parser.add_argument('--DG_weight', type=float, default=1.0, help='The loss weight for DG')
parser.add_argument('--mixtype', type=str, default='kde')
parser.add_argument('--kde_type', type=str, default='gaussian')
parser.add_argument('--kde_bandwidth', type=float, default=0.6)

# Augmentation
parser.add_argument('--use_aug', action='store_true', default=False, help='Set True to generate augmented samples in a batch')

# P-tuning
parser.add_argument('--P_token_num', type=int, default=30, help='the number of learnable tokens for P-tuning')
# LLM fine-tuning hyper-parameters
parser.add_argument('--use_LoRA', action='store_true', default=False, help='Set True to use LoRA')
parser.add_argument('--tune_layers', type=int, default=16, help='The number of last layers of LLM to tune')
parser.add_argument('--LoRA_r', type=int, default=8, help='r for LoRA')
parser.add_argument('--LoRA_dropOut', type=float, default=0.1, help='dropout rate for LoRA')

# Pretrain
parser.add_argument('--Pretrained_model_path', type=str, default='', help='The path to the saved pretrained model parameters')

# Ablation Study
parser.add_argument('--wo_DKPrompt', action='store_true', default=False, help='Set True to remove domain knowledge prompt')

# BatteryFormer
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')


args = parser.parse_args()

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)

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
    #     args.llm_layers, args.use_LoRA, args.lradj, args.dataset, args.use_align, args.use_DG, args.loss, args.wd, args.weighted_loss, args.wo_DKPrompt, pretrained, args.tune_layers)
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_pretrained{}_tl{}_rnnL{}_dr{}_DGW{}_Sig{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers, args.use_LoRA, args.lradj, args.dataset, args.use_align, args.use_DG, args.loss, args.wd, args.weighted_loss, pretrained, args.tune_layers, args.lstm_layers, args.dropout, args.DG_weight, args.kde_bandwidth)

    data_provider_func = data_provider_LLMv2
    if args.model == 'BatteryLifeLLMv7':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_FlattenHead':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_FlattenHead.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_GRUHead_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_GRUHead_tuneLLM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_MLPHead_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_MLPHead_tuneLLM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_pe':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_pe.Model(model_config)
    elif args.model == 'BatteryLifeLLMv13_SDCM_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv18_Stack.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_PTuning':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_PTuning.Model(model_config)
    elif args.model == 'BatteryLifeLLMv8_CP2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial.Model(model_config)
    elif args.model == 'BatteryLifeLLMv12_SDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv12_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_LinearHead2_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead2_tuneLLM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv8':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv8.Model(model_config)
    elif args.model == 'BatteryLifeLLMv13_SDCM_imp':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv13_SDCM_imp.Model(model_config)
    elif args.model == 'BatteryLifeLLMv8_S_reprogramming':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv8_S_reprogramming.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_LinearHead2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead2.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_LinearHead_aug':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead_aug.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_LinearHead_S':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead_S.Model(model_config)
    elif args.model == 'BatteryLifeLLMv13_SDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv13_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv8_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_LinearHead':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_FreConv':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_PSDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_MetaSDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_P.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_full_SDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_full_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_LLM_noTune_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_R.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv9.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_SDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_LLM_noTune_hyper_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_CM_R.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_LLM_noTune_hyper_Trial_P':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv9_LLM_noTune_hyper_Trial_P.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_ITCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_ITCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_selfD':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_selfD.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_ITSDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_ITSDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_Trial.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_TrialP2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2.Model(model_config)
    elif args.model == 'BatteryLifeLLMv11':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv11.Model(model_config)
    elif args.model == 'BatteryLifeLLMv11_SDCM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv11_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_Trial_noCycleP':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_Trial_noCycleP.Model(model_config)
    elif args.model == 'BatteryLifeLLMv10_Trial_noDKP':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_Trial_noDKP.Model(model_config)
    else:
        model = TimeLLM.Model(args)

    

    
    tokenizer = None
    if 'BatteryLifeLLM' in args.model:
        tokenizer = model.tokenizer
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    train_data, train_loader = data_provider_func(args, 'train', tokenizer, sample_weighted=args.weighted_sampling)
    label_scaler = train_data.return_label_scaler()        
    
    accelerator.print("Loading training samples......")
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, sample_weighted=args.weighted_sampling)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, sample_weighted=args.weighted_sampling)
    
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
        project="BatteryLifeLLM_final",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )
    # args.content = load_content(args)

    # load LoRA
    # print the module name



    if args.use_LoRA:
        # LLM_lora_config = LoraConfig(
        #     r=args.LoRA_r,
        #     lora_alpha=args.LoRA_r,
        #     lora_dropout=args.LoRA_dropOut,
        #     target_modules=["language_model.layers.31.self_attn.q_proj", "language_model.layers.31.self_attn.v_proj"],
        #     use_rslora=True, # sqrt(r)
        #     modules_to_save=['head1', 'cpl']
        # )

        set_off = args.llm_layers - args.tune_layers
        tune_layers = [i+set_off for i in range(args.tune_layers)]
        q_projs = [str(i) + '.self_attn.q_proj' for i in tune_layers]
        v_projs = [str(i) + '.self_attn.v_proj' for i in tune_layers]
        target_modules = q_projs + v_projs
        if '_P' in args.model:
            LLM_lora_config = LoraConfig(
                r=args.LoRA_r,
                lora_alpha=args.LoRA_r,
                lora_dropout=args.LoRA_dropOut,
                target_modules=target_modules,
                use_rslora=True, # sqrt(r)
                modules_to_save=['regression_head', 'regression_cpl', 'prompt_maker']
            )
        elif args.model == 'BatteryLifeLLMv10_Trial':
            LLM_lora_config = LoraConfig(
                r=args.LoRA_r,
                lora_alpha=args.LoRA_r,
                lora_dropout=args.LoRA_dropOut,
                target_modules=target_modules,
                use_rslora=True, # sqrt(r)
                modules_to_save=['regression_head', 'regression_cpl', 'prompt_fusion']
            )
        else:
            LLM_lora_config = LoraConfig(
                r=args.LoRA_r,
                lora_alpha=args.LoRA_r,
                lora_dropout=args.LoRA_dropOut,
                target_modules=target_modules,
                use_rslora=True, # sqrt(r)
                modules_to_save=['regression_head', 'regression_cpl']
            )

        model.add_adapter(LLM_lora_config)
        model = get_peft_model(model, LLM_lora_config)
        model.print_trainable_parameters()
    else:
        para_res = get_parameter_number(model)
        accelerator.print(para_res)
    
    for n, m in model.named_modules():
        # print the module name
        accelerator.print(n, m)

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
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    else:
        model_optim = optim.AdamW(trained_parameters, lr=args.learning_rate, weight_decay=args.wd)


    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    if args.loss != 'BMSE':
        criterion = nn.MSELoss(reduction='none') 
    else:
        criterion = bmc_loss
    

    prompt_adapter_loss = nn.CrossEntropyLoss()
    alignment_criterion = Alignment_loss(temperature=args.tau, instance_alingment_weight=args.beta)
    euclidean_dist = nn.PairwiseDistance(p=2)

    # accelerator.state.select_deepspeed_plugin("BatteryLifeLLM")
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    best_vali_loss = float('inf')
    best_vali_MAE, best_test_MAE = 0, 0
    best_vali_RMSE, best_test_RMSE = 0, 0
    for epoch in range(args.train_epochs):
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        total_alignment_loss = 0
        total_DG_loss = 0
        total_align2_loss = 0
        total_label_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_cl_loss = 0
        print_alignment_loss = 0
        print_DG_loss = 0
        print_align2_loss = 0
        print_label_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []
        for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, weights, end_input_ids, end_attn_mask, _, DKP_embeddings, cluster_labels) in enumerate(train_loader):
            with accelerator.accumulate(model):
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                model_optim.zero_grad()
                lambd = np.random.beta(1, 6) # dominant ratio of X2
                iter_count += 1
                
                cycle_curve_data = cycle_curve_data.float()
                curve_attn_mask = curve_attn_mask.float() # [B, L]
                DKP_embeddings = DKP_embeddings.float()
                cluster_labels = cluster_labels.long()
                labels = labels.float()
                # if 'v11' in args.model:
                #     if i % 2 == 0:
                #         # use adpater samples
                #         # adapter is frozen in this iteration
                #         labels = torch.cat([labels, labels], dim=0)

                input_ids = input_ids.int()
                attention_mask = attention_mask.int()
                weights = weights.float()
                end_input_ids = end_input_ids.int()
                end_attn_mask = end_attn_mask.int()
                


                # encoder - decoder
                outputs, prompt_scores, llm_out, feature_llm_out, _, _, _ = model(cycle_curve_data, curve_attn_mask, 
                input_ids=input_ids, attention_mask=attention_mask, end_input_ids=end_input_ids, end_attn_mask=end_attn_mask,
                DKP_embeddings=DKP_embeddings, cluster_labels=cluster_labels)
                if 'v11' in args.model or 'v13' in args.model:
                    if epoch <= args.noDG_epochs:
                        outputs = outputs[outputs.shape[0]//2:]
                    else:
                        labels = torch.cat([labels, labels], dim=0)
                cut_off = labels.shape[0]

                if args.loss == 'MSE':
                    loss = criterion(outputs[:cut_off], labels)
                    loss = torch.mean(loss * weights)
                elif args.loss == 'BMSE':
                    loss = criterion(outputs[:cut_off], labels, 1, False)
                    loss = torch.mean(loss * weights)
                elif args.loss == 'MAPE':
                    tmp_outputs = outputs[:cut_off] * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                    loss = torch.mean(loss * weights)

                final_loss = loss
                
                if args.use_DG and epoch > args.noDG_epochs:
                    # update the adapter
                    dg_loss = prompt_adapter_loss(prompt_scores, cluster_labels)
                    print_DG_loss = dg_loss.detach().float()
                    final_loss = loss + args.DG_weight * dg_loss
                    
                # if args.use_align:
                #     tmp_llm_out = llm_out[:cut_off]
                #     # tmp_aug_llm_out = llm_out[cut_off:]
                #     # alignment_loss, center_alignment_loss, instance_alignment_loss = alignment_criterion(tmp_llm_out, tmp_aug_llm_out, label_prompt_embedding)
                #     # print_alignment_loss = center_alignment_loss.detach().float()
                #     # print_align2_loss = instance_alignment_loss.detach().float()
                #     # print_cl_loss = alignment_loss.item()
                #     alignment_loss = torch.norm(tmp_llm_out - label_prompt_embedding, p=2, dim=-1) # [N]
                #     alignment_loss = torch.mean(alignment_loss)
                #     print_alignment_loss = alignment_loss.detach().float()

                #     final_loss = loss + args.gamma * alignment_loss

                    
                print_label_loss = loss.item()
                print_loss = final_loss.item()
                
                total_loss += final_loss.item()
                total_cl_loss += print_cl_loss
                total_alignment_loss += print_alignment_loss
                total_DG_loss += print_DG_loss
                total_align2_loss += print_align2_loss
                total_label_loss += print_label_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))

                accelerator.backward(final_loss)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # gradient clipping
                model_optim.step()
                

                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()


                
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {print_label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | align_loss: {print_alignment_loss:.7f} | align2_loss: {print_align2_loss:.7f} | DG loss: {print_DG_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                

        train_rmse = root_mean_squared_error(total_references, total_preds)
        train_mape = mean_absolute_percentage_error(total_references, total_preds)
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_rmse, vali_mae_loss, vali_mape = vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion)
        test_rmse, test_mae_loss, test_mape = vali_batteryLifeLLM(args, accelerator, model, test_data, test_loader, criterion)
        # test_rmse, vali_rmse = np.sqrt(test_loss), np.sqrt(vali_loss)
        # if args.loss == 'MAPE':
        #     vali_loss = vali_mape
        # elif args.loss == 'MSE':
        #     pass
        vali_loss = vali_mape

        
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            best_vali_MAE = vali_mae_loss
            best_test_MAE = test_mae_loss
            best_vali_RMSE = vali_rmse
            best_test_RMSE = test_rmse
            best_vali_MAPE = vali_mape
            best_test_MAPE = test_mape
            
        train_loss = total_loss / len(train_loader)
        total_cl_loss = total_cl_loss / len(train_loader)
        total_align2_loss = total_align2_loss / len(train_loader)
        total_alignment_loss = total_alignment_loss / len(train_loader)
        total_DG_loss = total_DG_loss / len(train_loader)
        total_label_loss = total_label_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train label loss: {total_label_loss:.5f} | Train cl loss: {total_cl_loss:.5f}| Train align2 loss: {total_align2_loss:.5f} | Train align loss: {total_alignment_loss:.5f} | Train DG loss: {total_DG_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali R{args.loss}: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test R{args.loss}: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_MAE": vali_mae_loss, "vali_R{args.loss}":vali_rmse, "test_MAE": test_mae_loss, "test_R{args.loss}": test_rmse})
        
        early_stopping(epoch+1, vali_loss, vali_mae_loss, test_mae_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            accelerator.set_trigger()
            
        if accelerator.check_trigger():
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                # if epoch == 0:
                #     args.learning_rate = model_optim.param_groups[0]['lr']
                #     accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test R{args.loss}: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Val MAE: {best_vali_MAE:.4f} | Val R{args.loss}: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f}')
accelerator.print(path)
accelerator.set_trigger()
if accelerator.check_trigger() and accelerator.is_local_main_process:
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_MAE": best_vali_MAE, f"vali_R{args.loss}": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "test_MAE": best_test_MAE, "test_R{args.loss}": best_test_RMSE, "test_MAPE":best_test_MAPE})
    wandb.finish()
