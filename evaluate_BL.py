import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
# from models import CPGRU, CPLSTM, CPMLP, CPBiGRU, CPBiLSTM, CPTransformer, PatchTST, iTransformer, Transformer, \
#     DLinear, Autoformer, MLP, MICN, CNN, \
#     BiLSTM, BiGRU, GRU, LSTM
from models import CPMLP_BL, CPTransformer_BL
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_evaluate_BL
import time
import random
import numpy as np
import os
import json
import datetime
import joblib
from utils.tools import EarlyStopping
from layers.Adapters import CPLayerWithAdapter, tLayerWithAdapter, CPTtLayerWithAdapter

parser = argparse.ArgumentParser(description='Time-LLM')

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

def calculate_metrics_based_on_seen_number_of_cycles(total_preds, total_references, total_seen_number_of_cycles, alpha1, alpha2, model, dataset, seed, trained_dataset, start=1, end=100):
    number_MAPE = {}
    number_alphaAcc1 = {}
    number_alphaAcc2 = {}
    for number in range(start, end+1):
        preds = total_preds[total_seen_number_of_cycles==number]
        references = total_references[total_seen_number_of_cycles==number]

        mape = mean_absolute_percentage_error(references, preds)
        relative_error = abs(preds - references) / references
        hit_num = sum(relative_error<=alpha)
        alpha_acc = hit_num / len(references) * 100

        relative_error = abs(preds - references) / references
        hit_num = sum(relative_error<=alpha2)
        alpha_acc2 = hit_num / len(references) * 100

        number_MAPE[number] = float(mape)
        number_alphaAcc1[number] = float(alpha_acc)
        number_alphaAcc2[number] = float(alpha_acc2)
    
    output_path = './output_path/'
    os.makedirs(output_path, exist_ok=True)
    with open(f'{output_path}number_MAPE_{model}_{dataset}_{trained_dataset}_{seed}.json', 'w') as f:
        json.dump(number_MAPE, f)
    with open(f'{output_path}number_alphaAcc1_{model}_{dataset}_{trained_dataset}_{seed}.json', 'w') as f:
        json.dump(number_alphaAcc1, f)
    with open(f'{output_path}number_alphaAcc2_{model}_{dataset}_{trained_dataset}_{seed}.json', 'w') as f:
        json.dump(number_alphaAcc2, f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def add_adapters_withCP(args, model, adapter_size=64):
    original_layer = model.intra_flatten
    model.intra_flatten = CPLayerWithAdapter(
        args,
        original_layer, 
        adapter_size=adapter_size
    )

    for i in range(model.e_layers):
        # add adapters to intra-cycle encoder layers
        original_layer = model.intra_MLP[i]
        model.intra_MLP[i] = tLayerWithAdapter(
            args,
            original_layer, 
            adapter_size=adapter_size
        )

    if args.model == 'CPMLP':
        for i in range(model.d_layers):
            # add adapters to inter-cycle encoder layers
            original_layer = model.inter_MLP[i]
            model.inter_MLP[i] = tLayerWithAdapter(
                args,
                original_layer, 
                adapter_size=adapter_size
            )
    elif args.model == 'CPTransformer':
        for i in range(model.d_layers):
            # add adapters to inter-cycle encoder layers
            original_layer = model.inter_TransformerEncoder[i]
            model.inter_TransformerEncoder[i] = CPTtLayerWithAdapter(
                args,
                original_layer, 
                adapter_size=adapter_size
            )

    return model

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
parser.add_argument('--data', type=str, required=False, default='Dataset_original', help='dataset type')
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
parser.add_argument('--save_evaluation_res', action='store_true', default=False, help='the True to save the results; Only effective when eval_cycle_min or eval_cycle_max is smaller than 0')
parser.add_argument('--alpha', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')
parser.add_argument('--args_path', type=str, help='the path to the pretrained model parameters')
parser.add_argument('--eval_dataset', type=str, help='the target dataset')
parser.add_argument('--eval_cycle_min', type=int, default=10, help='The lower bound for evaluation')
parser.add_argument('--eval_cycle_max', type=int, default=10, help='The upper bound for evaluation')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')

# transfer learning: AT, FT
parser.add_argument('--finetune_dataset', type=str, help='the target dataset for model finetuning')
parser.add_argument('--finetune_method', type=str, default=None, help='the fine-tuning method. [FT, EFT, AT]')
parser.add_argument('--adapter_size', type=int, default=32, help='the path to the pretrained model parameters')


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
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2_baseline.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# load from the saved path
alpha = args.alpha
alpha2 = args.alpha2
args_path = args.args_path
dataset = args.eval_dataset
finetune_method = args.finetune_method
model = args.model
adapter_size = args.adapter_size
args_json = json.load(open(f'{args_path}args.json'))
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args_json['model'] = model
args_json['alpha1'] = alpha
args_json['alpha2'] = alpha2
args_json['finetune_method'] = finetune_method
args_json['adapter_size'] = adapter_size
args.__dict__ = args_json
finetune_dataset = args.finetune_dataset if 'finetune_dataset' in args_json else 'None'
trained_dataset = args.dataset

for ii in range(args.itr):
    # setting record of experiments
    if finetune_method is not None:
        setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_seed{}_dataset{}_loss{}_wd{}_wl{}_fm{}_as{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.seed, finetune_dataset, args.loss, args.wd, args.weighted_loss, args.finetune_method, args.adapter_size)
    else:
        setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_lradj{}_dataset{}_loss{}_wd{}_wl{}'.format(
            args.model,
            args.seq_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.lradj, trained_dataset, args.loss, args.wd, args.weighted_loss)


    data_provider_func = data_provider_evaluate_BL
    print('model is :', args.model)
    if args.model == 'CPMLP':
        model = CPMLP_BL.Model(args).float()
    elif args.model == 'CPTransformer':
        model = CPTransformer_BL.Model(args).float()
    else:
        raise Exception(f'The {args.model} is not an implemented baseline!')

    label_scaler = joblib.load(f'{args_path}/label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    life_class_scaler = joblib.load(f'{args_path}/life_class_scaler')
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', None, label_scaler=label_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max, life_class_scaler=life_class_scaler)

    # load LoRA
    # print the module name
    for name, module in model._modules.items():
        print (name," : ",module)
    
    time_now = time.time()

    criterion = nn.MSELoss()
    accumulation_steps = args.accumulation_steps

    load_checkpoint_in_model(model, args_path) # load the saved parameters into model
    
    trained_parameters = []
    trained_parameters_names = []
    use_view_experts = True

    if finetune_method == 'FT':
        # free the general experts and tune other parameters
        for name, p in model.named_parameters():
            if p.requires_grad is True:
                trained_parameters_names.append(name)
                trained_parameters.append(p)
    elif finetune_method == 'AT':
        # adapter tuning
        model = add_adapters_withCP(args, model, args.adapter_size) # add adapters before and after that flattenIntra
        for name, p in model.named_parameters():
            # only tune the adapters + gate + head
            if 'adapter' in name or 'gate' in name or 'regression_head' in name:
                if p.requires_grad is True:
                    trained_parameters_names.append(name)
                    trained_parameters.append(p)
    else:
        raise Exception(f'{finetune_method} is not implemented!')
            
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    test_loader, model, model_optim = accelerator.prepare(test_loader, model, model_optim)
    accelerator.print(f'The model is {args.model}')
    accelerator.print(f'The sample size of testing set is {len(test_data)}')
    accelerator.print(f'load model from:\n {args_path}')
    accelerator.print(f'Model is loaded!')


    total_transformed_preds, total_transformed_labels, total_cycles, total_inputs = [], [], [], []
    sample_size = 0
    total_preds, total_references = [], []
    total_dataset_ids = []
    total_seen_unseen_ids = []
    total_seen_number_of_cycles = []
    total_domain_ids = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, dataset_ids, seen_unseen_ids, domain_ids) in tqdm(enumerate(test_loader)):
            
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)# [B, S, N]
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)
            seen_number_of_cycles = torch.sum(curve_attn_mask, dim=1) # [B]

            # encoder - decoder
            outputs = model(cycle_curve_data, curve_attn_mask)

            # self.accelerator.wait_for_everyone()
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            all_predictions, all_targets, dataset_ids, seen_unseen_ids, seen_number_of_cycles, domain_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, dataset_ids, seen_unseen_ids, seen_number_of_cycles, domain_ids))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            total_dataset_ids = total_dataset_ids + dataset_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_number_of_cycles = total_seen_number_of_cycles + seen_number_of_cycles.detach().cpu().numpy().reshape(-1).tolist()
            total_domain_ids = total_domain_ids + domain_ids.detach().cpu().numpy().reshape(-1).tolist()
    
    res_path='./results'
    save_res = {}
    save_res[dataset] = {}
    # accelerator.wait_for_everyone()
    accelerator.set_trigger()
    if accelerator.check_trigger():
        os.makedirs(res_path, exist_ok=True)
        total_dataset_ids = np.array(total_dataset_ids)
        total_domain_ids = np.array(total_domain_ids)
        total_references = np.array(total_references)
        total_seen_unseen_ids = np.array(total_seen_unseen_ids)
        total_seen_number_of_cycles = np.array(total_seen_number_of_cycles)
        total_preds = np.array(total_preds)


        relative_error = abs(total_preds - total_references) / total_references
        hit_num = sum(relative_error<=alpha2)
        alpha_acc2 = hit_num / len(total_references) * 100


        relative_error = abs(total_preds - total_references) / total_references
        hit_num = sum(relative_error<=alpha)
        alpha_acc = hit_num / len(total_references) * 100

        tmp_mapes = np.abs(total_preds-total_references) / total_references

        domain_average_MAPE = np.mean(domain_average(torch.tensor(total_domain_ids), torch.tensor(tmp_mapes)))
        save_res[dataset]['mapes'] = list(tmp_mapes)
        save_res[dataset]['total_references'] = list(total_references)
        save_res[dataset]['total_preds'] = list(total_preds)
        save_res[dataset]['total_seen_unseen_ids'] = list(total_seen_unseen_ids)
        save_res[dataset]['domain_ids'] = list(total_domain_ids)
        trained_seed = args_json['seed']
        model_name = args_json['model']
        with open(f'{res_path}/{model_name}_{dataset}_{trained_seed}.json', 'w') as f:
            json.dump(save_res, f)
        

        mape = mean_absolute_percentage_error(total_references, total_preds)

        accelerator.print(f'{dataset}-{args.seed} | Eval cycle: {eval_cycle_min}-{eval_cycle_max} | MAPE: {mape} | {alpha}-accuracy: {alpha_acc}% | {alpha2}-accuracy: {alpha_acc2}%')
        accelerator.print(f'{dataset}-{args.seed} | Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Domain average MAPE: {domain_average_MAPE}')
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([0])
        unseen_references = total_references[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([0])
        seen_preds = total_preds[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([1])
        unseen_preds = total_preds[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([1])

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

        if len(unseen_references)==0:
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc1}%')
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen {alpha2}-accuracy: {seen_alpha_acc2}%')
        else:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen MAPE: {seen_mape} | Unseen MAPE: {unseen_mape} | Seen {alpha}-accuracy: {seen_alpha_acc1}% | Unseen {alpha}-accuracy: {unseen_alpha_acc1}%')
            accelerator.print(f'Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Seen {alpha2}-accuracy: {seen_alpha_acc2}% | Unseen {alpha2}-accuracy: {unseen_alpha_acc2}%')
    
        if eval_cycle_min is None or eval_cycle_max is None:
            calculate_metrics_based_on_seen_number_of_cycles(total_preds, total_references, total_seen_number_of_cycles, alpha, alpha2, args.model, dataset, trained_dataset=trained_dataset, start=args.seq_len, end=args.early_cycle_threshold, seed=args.seed)