# curriculum learning
import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
import matplotlib
from utils.tools import train_model_course, get_parameter_number
from utils.losses import bmc_loss, Battery_life_alignment_CL_loss, DG_loss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryLifeLLMv10_TrialP2_full_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_CM_R, BatteryLifeLLMv10_TrialP2_noDKP_PSDCM, BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask, BatteryLifeLLMv10_TrialP2_noDKP_PTuning, BatteryLifeLLMv10_TrialP2_noDKP_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_P, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_R, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv13_SDCM_imp, BatteryLifeLLMv18_Stack, BatteryLifeLLMv6, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, BatteryLifeLLMv9, TimeLLM, \
    BatteryLifeLLMv5_distilling_version,BatteryLifeLLMv5_simpleLSTM,\
        BatteryLifeLLMv6_redescribe, BatteryLifeLLMv7,BatteryLifeLLMv7_LinearHead_aug, BatteryLifeLLMv7_LinearHead2_tuneLLM, BatteryLifeLLMv7_GRUHead_tuneLLM, \
            BatteryLifeLLMv7_MLPHead_tuneLLM, BatteryLifeLLMv7_TransHead, \
            BatteryLifeLLMv7_LinearHead_S, BatteryLifeLLMv8_S_reprogramming, BatteryLifeLLMv9_LLM_noTune_hyper_Trial_P
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_LLMv2
import time
import random
import torch.nn.functional as F
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import datetime
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import json
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content, vali_batteryLifeLLM_relative_absolute_alignment
# wandb.login(key="90b2c598dc4a58105fdcdd8c03ec271984ec7417")
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
parser.add_argument('--neg_threshold', type=float, default=0.25)
parser.add_argument('--pos_threshold', type=float, default=0.15)
parser.add_argument('--neg_num', type=int, default=2)
parser.add_argument('--pos_num', type=int, default=1)
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--beta', type=float, default=1, help='Alignment weight')
parser.add_argument('--use_align', action='store_true', default=False, help='Set True to use contrastive learning')

# Deal with long-tail issue
parser.add_argument('--use_hyper', action='store_true', default=False, help='Set True to use HyperNN')
parser.add_argument('--life_class_weight', type=float, default=0.5)
# Domain generalization
parser.add_argument('--use_DG', action='store_true', default=False, help='Set True to use domain generalization')
parser.add_argument('--DG_weight', type=float, default=0.5)
# Augmentation
parser.add_argument('--use_aug', action='store_true', default=False, help='Set True to generate augmented samples in a batch')

# LLM fine-tuning hyper-parameters
parser.add_argument('--use_LoRA', action='store_true', default=False, help='Set True to use LoRA')
parser.add_argument('--tune_layers', type=int, default=16, help='The number of last layers of LLM to tune')
parser.add_argument('--LoRA_r', type=int, default=8, help='r for LoRA')
parser.add_argument('--LoRA_dropOut', type=float, default=0.0, help='dropout rate for LoRA')

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
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)

train_epochs = args.train_epochs
least_epochs = args.least_epochs
patience = args.patience
learning_rate = args.learning_rate
dropout = args.dropout
dataset = args.dataset
beta = args.beta
LLM_path = args.LLM_path
root_path = args.root_path
lradj = args.lradj
batch_size = args.batch_size
checkpoints = args.checkpoints
Pretrained_model_path = args.Pretrained_model_path
args_json = json.load(open(f'{Pretrained_model_path}args.json'))
args_json['train_epochs'] = train_epochs
args_json['least_epochs'] = least_epochs
args_json['patience'] = patience
args_json['learning_rate'] = learning_rate
args_json['LLM_path'] = LLM_path
args_json['dropout'] = dropout
args_json['root_path'] = root_path
args_json['beta'] = beta
args_json['lradj'] = lradj
args_json['checkpoints'] = checkpoints
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args.__dict__ = args_json
    
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_woDKPr{}_pretrained{}_tl{}_Stage1'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers, args.use_LoRA, args.lradj, args.dataset, args.use_align, args.use_DG, args.loss, args.wd, args.weighted_loss, args.wo_DKPrompt, False, args.tune_layers)


    data_provider_func = data_provider_LLMv2
    if args.model == 'BatteryLifeLLMv5_distilling_version':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv5_distilling_version.Model(model_config)
    elif args.model == 'BatteryLifeLLMv5_simpleLSTM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv5_simpleLSTM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv6_redescribe':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv6_redescribe.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7':
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
    elif args.model == 'BatteryLifeLLMv7_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv18_Stack.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_TransHead':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_TransHead.Model(model_config)
    elif args.model == 'BatteryLifeLLMv8_CP2':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial.Model(model_config)
    elif args.model == 'BatteryLifeLLMv7_MLPHead':
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
    elif args.model == 'BatteryLifeLLMv8_S':
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
    elif args.model == 'BatteryLifeLLMv7_decompose':
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
    elif args.model == 'BatteryLifeLLMv9_S':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_S_noTune':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_P.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_LLM_noTune':
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
    elif args.model == 'BatteryLifeLLMv9_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_PTuning.Model(model_config)
    elif args.model == 'BatteryLifeLLMv9_LLM_noTune_hyper':
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
    elif args.model == 'BatteryLifeLLMv9_Trial_P':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_PSDCM.Model(model_config)
    else:
        model = TimeLLM.Model(args).float()
        
    tokenizer = None
    if 'BatteryLifeLLM' in args.model:
        tokenizer = model.tokenizer
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    train_data, train_loader = data_provider_func(args, 'train', tokenizer, sample_weighted=args.weighted_sampling)
    label_scaler = train_data.return_label_scaler()  
    life_class_scaler = train_data.return_life_class_scaler()      
    
    accelerator.print("Loading training samples......")
    # accelerator.print("Loading vali samples......")
    # vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    # accelerator.print("Loading test samples......")
    # test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    
    if accelerator.is_local_main_process and os.path.exists(path):
        del_files(path)  # delete checkpoint files
        accelerator.print(f'success delete {path}')
    
    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{path}/label_scaler')
    joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    if accelerator.is_local_main_process:
        wandb.init(
        # set the wandb project where this run will be logged
        project="BatteryLifeLLM_Pretrain",
        
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
        # tune_layers = [i+8 for i in range(16)]
        set_off = args.llm_layers - args.tune_layers
        tune_layers = [i+set_off for i in range(args.tune_layers)]
        q_projs = [str(i) + '.self_attn.q_proj' for i in tune_layers]
        v_projs = [str(i) + '.self_attn.v_proj' for i in tune_layers]
        target_modules = q_projs + v_projs
        LLM_lora_config = LoraConfig(
            r=args.LoRA_r,
            lora_alpha=args.LoRA_r,
            lora_dropout=args.LoRA_dropOut,
            target_modules=target_modules,
            use_rslora=True, # sqrt(r)
            modules_to_save=['label_head', 'prompt_maker']
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
        if 'regression_cpl' in name:
            p.requires_grad = True
            trained_parameters_names.append(name)
            trained_parameters.append(p)
        else:
            p.requires_grad = False

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
    

    DG_criterion = DG_loss()
    life_class_criterion = nn.CrossEntropyLoss()
    euclidean_dist = nn.PairwiseDistance(p=2)

    train_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, model, model_optim, scheduler)


    best_vali_loss = float('inf')
    best_vali_MAE, best_test_MAE = 0, 0
    best_vali_RMSE, best_test_RMSE = 0, 0

    for epoch in range(1):
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        total_alignment_loss = 0
        total_DG_loss = 0
        total_align2_loss = 0
        total_label_loss = 0
        
        model.eval()
        epoch_time = time.time()
        print_cl_loss = 0
        print_alignment_loss = 0
        print_DG_loss = 0
        print_align2_loss = 0
        print_label_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_labels, total_features = [], []
        total_label_feature_out = []
        total_preds, total_references = [], []
        total_file_names = []
        with torch.no_grad():
            for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights, end_input_ids, end_attn_mask, label_prompt_embedding, label_input_ids, label_attention_mask, file_names) in enumerate(train_loader):
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                model_optim.zero_grad()
                iter_count += 1
                
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
                if args.use_DG:
                    # data augmentation is used
                    # prepare inputs
                    cj_aug_cycle_curve_data = cj_aug_cycle_curve_data.float()
                    cycle_curve_data = torch.cat([cycle_curve_data, cj_aug_cycle_curve_data], dim=0)
                    curve_attn_mask = torch.cat([curve_attn_mask, curve_attn_mask], dim=0)
                    input_ids = torch.cat([input_ids, input_ids], dim=0)
                    end_input_ids = torch.cat([end_input_ids, end_input_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                    end_attn_mask = torch.cat([end_attn_mask, end_attn_mask], dim=0)
                    cut_off = labels.shape[0]
                else:
                    cut_off = labels.shape[0]

                # encoder - decoder
                _, preds_life_class, llm_out, feature_llm_out, outputs, label_feature_llm_out, label_llm_out = model(cycle_curve_data, curve_attn_mask, 
                input_ids=input_ids, attention_mask=attention_mask, 
                end_input_ids=end_input_ids, end_attn_mask=end_attn_mask, 
                label_input_ids=label_input_ids, label_attention_mask=label_attention_mask)

                transformed_labels = labels * std + mean_value
                tmp_file_names = accelerator.gather_for_metrics((file_names))
                tmp_labels, tmp_features, tmp_label_feature_llm_out = accelerator.gather_for_metrics((transformed_labels, label_llm_out, label_feature_llm_out))
                total_file_names = total_file_names + tmp_file_names
                total_labels = total_labels + tmp_labels.float().detach().cpu().numpy().reshape(-1).tolist()
                total_features.append(tmp_features.float().detach().cpu().numpy())
                total_label_feature_out.append(tmp_label_feature_llm_out.float().detach().cpu().numpy())

        total_labels = np.array(total_labels)
        total_features = np.concatenate(total_features, axis=0)
        total_label_feature_out = np.concatenate(total_label_feature_out, axis=0)
        tsne = PCA(n_components=2, random_state=2024)
        transformed_features = tsne.fit_transform(total_features) # [N, 2]
        transformed_total_label_feature_out = tsne.fit_transform(total_label_feature_out) # [N, 2]

        colormap =  matplotlib.colormaps['coolwarm']
        norm = matplotlib.colors.Normalize(vmin=min(total_labels), vmax=max(total_labels))
        total_colors = MinMaxScaler().fit_transform(np.array(total_labels).reshape(-1, 1))
        total_colors = total_colors.reshape(-1)

        fig = plt.figure(figsize=(5,5))
        plt.scatter(transformed_features[:,0], transformed_features[:,1], c=colormap(total_colors))
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),label='Cycle life',ax=plt.gca())
        plt.title('label_llm_out')
        plt.savefig('label_llm_out.jpg', dpi=600)
        plt.show()

        fig = plt.figure(figsize=(5,5))
        plt.scatter(transformed_total_label_feature_out[:,0], transformed_total_label_feature_out[:,1], c=colormap(total_colors))
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),label='Cycle life',ax=plt.gca())
        plt.title('label_feature_llm_out')
        plt.savefig('label_feature_llm_out.jpg', dpi=600)
        plt.show()

        np.save(f"{Pretrained_model_path}total_label_llm_out.npy",total_features)
        np.save(f"{Pretrained_model_path}total_label_feature_llm_out.npy",total_label_feature_out)
        np.save(f"{Pretrained_model_path}total_labels.npy",total_labels)
        with open(f'{Pretrained_model_path}total_file_names.json', 'w') as f:
            json.dump(total_file_names, f)
