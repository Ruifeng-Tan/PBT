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
from models import PBT, baseline_CPTransformerMoE, baseline_CPMLPMoE, CPTransformer_ablation, CPTransformer, CPMLP
import wandb
from data_provider.gate_masker import gate_masker
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_LLMv2, data_provider_LLM_evaluate
import time
import random
import numpy as np
import os
import json
import datetime
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
import joblib
from utils.tools import del_files, EarlyStopping, domain_average, vali_batteryLifeLLM
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
parser.add_argument('--pt_token_num', type=int, default=10, help='The token number for prompt tuning')
parser.add_argument('--last_layer', type=int, default=0, help='The layer index for fusion')
parser.add_argument('--d_llm', type=int, default=4096, help='the features of llm')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--noDKP_layers', type=int, default=1, help='the number of no DKP layers in the inter-cycle encoder')
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
parser.add_argument('--lradj_factor', type=float, default=0.5, help='the learning rate decay factor')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--top_p', type=float, default=0.5, help='The threshold used to control the number of activated experts')
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)

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
parser.add_argument('--initial_lambda', type=float, default=1e-4, help='The initial lambda for relu router regularization')
parser.add_argument('--initial_alpha', type=float, default=1.2, help='The initial alpha for relu router regularization')

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
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero_ours.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.accumulation_steps)
# load from the saved path
args_path = args.args_path
dataset = args.eval_dataset
alpha = args.alpha1
alpha2 = args.alpha2
args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size

args.__dict__ = args_json
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_le{}_bs{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_mdf{}_lradj{}_{}_guide{}_LB{}_loss{}_wd{}_wl{}_dr{}_gdff{}_E{}_GE{}_K{}_S{}_aug{}_augW{}_tem{}_wDG{}_dsr{}_we{}_ffs{}_seed{}'.format(
        args.model,
        args.dk_factor,
        args.llm_choice,
        args.seq_len,
        args.least_epochs,
        args.batch_size,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.min_d_ff,
        args.lradj, trained_dataset, args.use_guide, args.use_LB, args.loss, args.wd, args.weighted_loss, args.dropout, args.gate_d_ff, 
        args.num_experts, args.num_general_experts,
        args.topK, args.use_domainSampler, args.use_aug, args.aug_w, args.temperature, args.weighted_CLDG, args.down_sample_ratio, args.warm_up_epoches, args.use_dff_scale, args.seed)


    data_provider_func = data_provider_LLM_evaluate
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
    elif args.model == 'PBT':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = PBT.Model(model_config)
    elif args.model == 'CPTransformer_ablation':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = CPTransformer_ablation.Model(model_config)
    else:
        raise Exception('Not Implemented')

    
    path = args_path  # unique checkpoint saving path
    
    tokenizer = None
    if model.tokenizer:
        tokenizer = model.tokenizer

    if not 'MIX_all' in trained_dataset:
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

    label_scaler = joblib.load(f'{path}label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    accelerator.print("Loading training samples......")
    # accelerator.print("Loading vali samples......")
    # vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, 
                                                label_scaler=label_scaler, eval_cycle_min=eval_cycle_min, 
                                                eval_cycle_max=eval_cycle_max, temperature2mask=temperature2mask, format2mask=format2mask, cathodes2mask=cathodes2mask, anode2mask=anode2mask, ion2mask=ion2mask, trained_dataset=trained_dataset)


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
    total_domain_ids = []
    total_seen_unseen_ids = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask, labels, weights, dataset_ids, seen_unseen_ids, DKP_embeddings, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids) in tqdm(enumerate(test_loader)):
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

            # encoder - decoder
            outputs, prompt_scores, llm_out, feature_llm_out, _, alpha_exponent, aug_loss, guide_loss = model(cycle_curve_data, curve_attn_mask, 
            DKP_embeddings=DKP_embeddings, cathode_masks=cathode_masks, temperature_masks=temperature_masks, format_masks=format_masks, 
            anode_masks=anode_masks, ion_type_masks=ion_type_masks, combined_masks=combined_masks)
            # self.accelerator.wait_for_everyone()
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            all_predictions, all_targets, dataset_ids, seen_unseen_ids, domain_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, dataset_ids, seen_unseen_ids, domain_ids))
            
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_domain_ids = total_domain_ids + domain_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            total_dataset_ids = total_dataset_ids + dataset_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()

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

        accelerator.print(f'{dataset} | Eval cycle: {eval_cycle_min}-{eval_cycle_max} | MAPE: {mape} | {alpha}-accuracy: {alpha_acc}% | {alpha2}-accuracy: {alpha_acc2}%')
        accelerator.print(f'{dataset} | Eval cycle: {eval_cycle_min}-{eval_cycle_max} | Domain average MAPE: {domain_average_MAPE}')
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




            