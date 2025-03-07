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
from utils.tools import train_model_course, get_parameter_number
from utils.losses import bmc_loss, Battery_life_alignment_CL_loss, DG_loss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryLifeLLMv10_TrialP2_full_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_CM_R, BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask, BatteryLifeLLMv10_TrialP2_noDKP_PTuning, BatteryLifeLLMv10_TrialP2_noDKP_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_P, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_R, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv13_SDCM_imp, BatteryLifeLLMv18_Stack, BatteryLifeLLMv6, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, BatteryLifeLLMv9, TimeLLM, \
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
import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import json
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content
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

if args.Pretrained_model_path:
    pretrained = True
else:
    pretrained = False
    
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_woDKPr{}_pretrained{}_tl{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers, args.use_LoRA, args.lradj, args.dataset, args.use_align, args.use_DG, args.loss, args.wd, args.weighted_loss, args.wo_DKPrompt, pretrained, args.tune_layers)


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
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    
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
        LLM_lora_config = LoraConfig(
            r=args.LoRA_r,
            lora_alpha=args.LoRA_r,
            lora_dropout=args.LoRA_dropOut,
            target_modules=target_modules,
            use_rslora=True, # sqrt(r)
            modules_to_save=['cpl']
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

    life_classes = json.load(open('data_provider/life_classes.json'))
    class_numbers = len(list(life_classes.keys()))
    if args.loss != 'BMSE':
        criterion = nn.MSELoss(reduction='none') 
    else:
        criterion = bmc_loss
    

    DG_criterion = DG_loss()
    life_class_criterion = nn.CrossEntropyLoss()
    euclidean_dist = nn.PairwiseDistance(p=2)

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
        for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights, class_centers, end_input_ids, end_attn_mask, label_prompt_embedding) in enumerate(train_loader):
            with accelerator.accumulate(model):
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                model_optim.zero_grad()
                iter_count += 1
                
                life_class = life_class.to(accelerator.device)
                scaled_life_class = scaled_life_class.long().to(accelerator.device)
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                labels = labels.float().to(accelerator.device)
                input_ids = input_ids.int().to(accelerator.device)
                attention_mask = attention_mask.int().to(accelerator.device)
                weights = weights.float().to(accelerator.device)
                end_input_ids = end_input_ids.int().to(accelerator.device)
                end_attn_mask = end_attn_mask.int().to(accelerator.device)
                label_prompt_embedding = label_prompt_embedding.float().to(accelerator.device)
                if args.use_DG:
                    # data augmentation is used
                    # prepare inputs
                    class_centers = class_centers.float().to(accelerator.device)
                    cj_aug_cycle_curve_data = cj_aug_cycle_curve_data.float().to(accelerator.device)
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
                outputs, features, cl_embed, preds_life_class, llm_out = model(cycle_curve_data, curve_attn_mask, 
                input_ids=input_ids, attention_mask=attention_mask, end_input_ids=end_input_ids, end_attn_mask=end_attn_mask)


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

                
                
                # if args.use_hyper:
                #     pass
                #     # life_class_loss = life_class_criterion(preds_life_class, scaled_life_class)
                #     # print_align2_loss = life_class_loss.detach().float()
                #     # final_loss = final_loss + args.life_class_weight * life_class_loss
                D = label_prompt_embedding.shape[-1] # the d_llm
                if args.use_DG:
                    dg_loss = DG_criterion(llm_out, scaled_life_class, class_centers)
                    print_DG_loss = dg_loss.item()
                    final_loss = loss + args.DG_weight * dg_loss
                elif args.use_align:
                    tmp_llm_out = llm_out[:cut_off]
                    # minimize the cosine similarit between the embeddings of the learned prompt and label prompt
                    cosine_sim = torch.cosine_similarity(tmp_llm_out, label_prompt_embedding, dim=1)
                    align_loss = - torch.mean(cosine_sim)
                    print_alignment_loss = align_loss.detach().float()
                    distances = euclidean_dist(tmp_llm_out, label_prompt_embedding) # [N]
                    # distances = torch.norm(tmp_llm_out-label_llm_out, p=2, dim=1) # [N]
                    align_loss2 = torch.mean(distances)  / np.sqrt(D)
                    # align_loss2 = F.relu(align_loss2-0.25)
                    print_align2_loss = align_loss2.detach().float()

                    # contrastive learning
                    distance_matrix = torch.norm(tmp_llm_out.unsqueeze(1)-label_prompt_embedding.unsqueeze(0), p=2, dim=-1) / np.sqrt(D) / args.tau # [N, N]
                    sim_matrix = torch.sum(torch.exp(distance_matrix), dim=1) # [N]

                    distances = distances / np.sqrt(D) / args.tau
                    pos_dist = torch.exp(distances) # [N]
                    cl_loss = torch.mean(torch.log(pos_dist/sim_matrix))
                    print_cl_loss = cl_loss.item()

                    final_loss = loss + args.beta * align_loss2
                else:
                    tmp_llm_out = llm_out[:cut_off]
                    # minimize the cosine similarit between the embeddings of the learned prompt and label prompt
                    cosine_sim = torch.cosine_similarity(tmp_llm_out, label_prompt_embedding, dim=1)
                    align_loss = - torch.mean(cosine_sim)
                    print_alignment_loss = align_loss.item()
                    distances = euclidean_dist(tmp_llm_out, label_prompt_embedding)
                    # distances = torch.norm(tmp_llm_out-label_llm_out, p=2, dim=1) # [N]
                    align_loss2 = torch.mean(distances) / np.sqrt(D)
                    print_align2_loss = align_loss2.item()
                    # align_loss = align_loss + 0.01 * align_loss2
                    align_loss = align_loss2
                    final_loss = loss

                    
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

                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
                accelerator.backward(final_loss)
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # gradient clipping
                model_optim.step()


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
