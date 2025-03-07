import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer
from transformers import AutoConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask, BatteryLifeLLMv10_TrialP2_noDKP_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv13_SDCM_imp, BatteryLifeLLMv18_Stack, BatteryLifeLLMv6, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, TimeLLM, \
    BatteryLifeLLMv5_distilling_version,BatteryLifeLLMv5_simpleLSTM,\
        BatteryLifeLLMv6_redescribe, BatteryLifeLLMv7,BatteryLifeLLMv7_LinearHead_aug, BatteryLifeLLMv7_LinearHead2_tuneLLM, BatteryLifeLLMv7_GRUHead_tuneLLM, \
            BatteryLifeLLMv7_MLPHead_tuneLLM, BatteryLifeLLMv7_TransHead, BatteryLifeLLMv7_LinearHead_S, \
                BatteryLifeLLMv8_S_reprogramming, LSTM, GRU, Transformer
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_LLMv2, data_provider_LLM_evaluate
import time
import random
import numpy as np
import os
import json
import datetime
from data_provider.data_factory import data_provider_baseline
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
import joblib
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content
parser = argparse.ArgumentParser(description='Time-LLM')

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
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)

# Contrastive learning
parser.add_argument('--neg_threshold', type=float, default=0.25)
parser.add_argument('--pos_threshold', type=float, default=0.15)
parser.add_argument('--neg_num', type=int, default=2)
parser.add_argument('--pos_num', type=int, default=1)
parser.add_argument('--tau', type=int, default=1)
parser.add_argument('--cl_loss_weight', type=float, default=1)

# LLM fine-tuning hyper-parameters
parser.add_argument('--use_LoRA', action='store_true', default=False, help='Set True to use LoRA')
parser.add_argument('--LoRA_r', type=int, default=8, help='r for LoRA')
parser.add_argument('--LoRA_dropOut', type=float, default=0.1, help='dropout rate for LoRA')

# BatteryFormer
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')

# evaluate
parser.add_argument('--alpha', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')
parser.add_argument('--args_path', type=str, help='the path to the pretrained model parameters')
parser.add_argument('--eval_dataset', type=str, help='the target dataset')
parser.add_argument('--eval_cycle_min', type=int, default=10, help='The lower bound for evaluation')
parser.add_argument('--eval_cycle_max', type=int, default=10, help='The upper bound for evaluation')
args = parser.parse_args()
eval_cycle_min = args.eval_cycle_min
eval_cycle_max = args.eval_cycle_max
batch_size = args.batch_size
if eval_cycle_min < 0 or eval_cycle_max <0:
    eval_cycle_min = None
    eval_cycle_max = None
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# load from the saved path
args_path = args.args_path
dataset = args.eval_dataset
alpha = args.alpha
alpha2 = args.alpha2
args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args.__dict__ = args_json
for ii in range(args.itr):
    # setting record of experiments
    # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_llmLayers{}_{}_{}_Lora{}'.format(
    #     args.task_name,
    #     args.model_id,
    #     args.model,
    #     args.data,
    #     args.features,
    #     args.seq_len,
    #     args.label_len,
    #     args.d_model,
    #     args.n_heads,
    #     args.e_layers,
    #     args.d_layers,
    #     args.d_ff,
    #     args.factor,
    #     args.embed,
    #     args.llm_layers,
    #     args.des, args.use_LoRA, ii)
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_hyper{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers, args.use_LoRA, args.lradj, trained_dataset, args.use_align, args.DG, args.loss, args.wd, args.weighted_loss, args.use_hyper)


    data_provider_func = data_provider_LLM_evaluate
    if args.model == 'LSTM':
        model = LSTM.Model(args)
    elif args.model == 'GRU':
        model = GRU.Model(args)
    elif args.model == 'Transformer':
        model = Transformer.Model(args)
    else:
        raise Exception('Not implemented')
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    tokenizer = AutoTokenizer.from_pretrained(
            'deepset/sentence_bert',
            # 'huggyllama/llama-7b',
            trust_remote_code=True
        ) # The ouput of the tokenizer won't be used. We just randomly assign a tokenizer here.
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    label_scaler = joblib.load(f'{path}/label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    life_class_scaler = joblib.load(f'{path}/life_class_scaler')
    accelerator.print("Loading training samples......")
    # accelerator.print("Loading vali samples......")
    # vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler=label_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max, life_class_scaler=life_class_scaler)


    # load LoRA
    # print the module name
    for name, module in model._modules.items():
        print (name," : ",module)
        
        
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
            
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    time_now = time.time()

    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    criterion = nn.MSELoss()
    accumulation_steps = args.accumulation_steps
    load_checkpoint_in_model(model, path) # load the saved parameters into model
    test_loader, model, model_optim = accelerator.prepare(test_loader, model, model_optim)
    accelerator.print(f'The model is {args.model}')
    accelerator.print(f'The sample size of testing set is {len(test_data)}')
    accelerator.print(f'load model from:\n {path}')
    # accelerator.load_checkpoint_in_model(model, path) # load the saved parameters into model
    accelerator.print(f'Model is loaded!')


    total_transformed_preds, total_transformed_labels, total_cycles, total_inputs = [], [], [], []
    sample_size = 0
    total_preds, total_references = [], []
    total_dataset_ids = []
    total_seen_unseen_ids = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class, weights, dataset_ids, seen_unseen_ids) in tqdm(enumerate(test_loader)):
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)# [B, S, N]
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)

            # encoder - decoder
            outputs = model(cycle_curve_data, curve_attn_mask)
            # self.accelerator.wait_for_everyone()
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            all_predictions, all_targets, dataset_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, dataset_ids))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            total_dataset_ids = total_dataset_ids + dataset_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()

    res_path='./results'
    # accelerator.wait_for_everyone()
    accelerator.set_trigger()
    if accelerator.check_trigger():
        total_dataset_ids = np.array(total_dataset_ids)
        total_references = np.array(total_references)
        total_seen_unseen_ids = np.array(total_seen_unseen_ids)
        total_preds = np.array(total_preds)

        relative_error = abs(total_preds - total_references) / total_references
        hit_num = sum(relative_error<=alpha)
        alpha_acc = hit_num / len(total_references) * 100

        relative_error = abs(total_preds - total_references) / total_references
        hit_num = sum(relative_error<=alpha2)
        alpha_acc2 = hit_num / len(total_references) * 100


        relative_error = abs(total_preds - total_references) / total_references
        hit_num = sum(relative_error<=alpha)
        alpha_acc = hit_num / len(total_references) * 100


        mape = mean_absolute_percentage_error(total_references, total_preds)

        accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | MAPE: {mape} | {alpha}-accuracy: {alpha_acc}% | {alpha2}-accuracy: {alpha_acc2}%')
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1]
        unseen_references = total_references[total_seen_unseen_ids==0]
        seen_preds = total_preds[total_seen_unseen_ids==1]
        unseen_preds = total_preds[total_seen_unseen_ids==0]

        seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=alpha)
        seen_alpha_acc = hit_num / len(seen_references) * 100

        if len(unseen_references) == 0:
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc}%')
        else:
            unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=alpha)
            unseen_alpha_acc = hit_num / len(unseen_references) * 100
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Unseen MAPE: {unseen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc}% | Unseen {alpha}-accuracy: {unseen_alpha_acc}%')

        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=alpha2)
        seen_alpha_acc2 = hit_num / len(seen_references) * 100

        if len(unseen_references)==0:
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc}%')
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen {alpha2}-accuracy: {seen_alpha_acc2}%')
        else:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Unseen MAPE: {unseen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc}% | Unseen {alpha}-accuracy: {unseen_alpha_acc}%')
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen {alpha2}-accuracy: {seen_alpha_acc2}% | Unseen {alpha2}-accuracy: {unseen_alpha_acc2}%')




            