import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
from transformers import AutoConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import BatteryLifeLLMv10_TrialP2_noDKP_CM_R, BatteryLifeLLMv10_TrialP2_noDKP_PTnoMask, BatteryLifeLLMv10_TrialP2_noDKP_SDCM, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv13_SDCM_imp, BatteryLifeLLMv18_Stack, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, TimeLLM, \
            BatteryLifeLLMv7,BatteryLifeLLMv7_LinearHead_aug, BatteryLifeLLMv7_LinearHead2_tuneLLM, BatteryLifeLLMv7_GRUHead_tuneLLM, \
            BatteryLifeLLMv7_MLPHead_tuneLLM, BatteryLifeLLMv7_TransHead, BatteryLifeLLMv7_LinearHead_S, \
                BatteryLifeLLMv8_S_reprogramming, LSTM, GRU, Transformer, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, \
                BatteryLifeLLMv9_LLM_noTune_hyper_Trial_P, BatteryLifeLLMv10_Trial, BatteryLifeLLMv11, BatteryLifeLLMv20_Stack_Trial
import wandb
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
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content
parser = argparse.ArgumentParser(description='Time-LLM')
def calculate_metrics_based_on_seen_number_of_cycles(total_preds, total_references, total_seen_number_of_cycles, alpha1, alpha2, model, dataset, seed, finetune_dataset, start=1, end=100):
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
    with open(f'{output_path}number_MAPE_{model}_{dataset}_{finetune_dataset}_{seed}.json', 'w') as f:
        json.dump(number_MAPE, f)
    with open(f'{output_path}number_alphaAcc1_{model}_{dataset}_{finetune_dataset}_{seed}.json', 'w') as f:
        json.dump(number_alphaAcc1, f)
    with open(f'{output_path}number_alphaAcc2_{model}_{dataset}_{finetune_dataset}_{seed}.json', 'w') as f:
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

# evaluate
parser.add_argument('--alpha', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--args_path', type=str, help='the path to the pretrained model parameters')
parser.add_argument('--LLM_path', type=str, help='the path to the LLM')
parser.add_argument('--root_path', type=str, help='the path to the dataset')
parser.add_argument('--eval_dataset', type=str, help='the target dataset')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--eval_cycle_min', type=int, default=10, help='The lower bound for evaluation')
parser.add_argument('--eval_cycle_max', type=int, default=10, help='The upper bound for evaluation')
args = parser.parse_args()
eval_cycle_min = args.eval_cycle_min
eval_cycle_max = args.eval_cycle_max
batch_size = args.batch_size
root_path = args.root_path
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
LLM_path = args.LLM_path
args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args_json['args_path'] = args_path
args_json['root_path'] = root_path
args_json['LLM_path'] = LLM_path
args.__dict__ = args_json
finetune_dataset = args.finetune_dataset if 'finetune_dataset' in args_json else 'None'

if args.Pretrained_model_path:
    pretrained = True
else:
    pretrained = False

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
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_Lora{}_lradj{}_dataset{}_align{}_DG{}_loss{}_wd{}_wl{}_pretrained{}_rnnL{}_dr{}_IW{}_NumE{}_K{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers, args.use_LoRA, args.lradj, trained_dataset, args.use_align, args.use_DG, args.loss, args.wd, args.weighted_loss, pretrained, args.lstm_layers, args.dropout, args.importance_weight, args.num_experts, args.topK)


    data_provider_func = data_provider_LLM_evaluate
    if args.model == 'BatteryLifeLLMv7':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7.Model(model_config)
    elif args.model == 'BatteryLifeLLMv20_Stack_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv20_Stack_Trial.Model(model_config)
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
    elif args.model == 'BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr':
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
    elif args.model == 'BatteryLifeLLMv10_Trial':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_Trial.Model(model_config)
    elif args.model == 'BatteryLifeLLMv11':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv11.Model(model_config)
    else:
        raise Exception(f'Please add {args.model} to the if-else')
    tokenizer = None
    if 'BatteryLifeLLM' in args.model:
        tokenizer = model.tokenizer
    
    path = os.path.join(args.args_path)  # unique checkpoint saving path
    
    label_scaler = joblib.load(f'{path}/label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]

    accelerator.print("Loading training samples......")
    # accelerator.print("Loading vali samples......")
    # vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler=label_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max)


    # load LoRA
    # print the module name
    for name, module in model._modules.items():
        print (name," : ",module)
    if args.use_LoRA:
        LLM_lora_config = LoraConfig(
            r=args.LoRA_r,
            lora_alpha=args.LoRA_r,
            lora_dropout=args.LoRA_dropOut,
            target_modules=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
            use_rslora=True, # sqrt(r)
            bias="none",
            modules_to_save=['head1', 'cpl']
        )
        model.add_adapter(LLM_lora_config)
        model = get_peft_model(model, LLM_lora_config)
        model.print_trainable_parameters()
        
        
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
            
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    time_now = time.time()

    early_stopping = EarlyStopping(args, accelerator=accelerator, patience=args.patience, least_epochs=args.least_epochs)

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
    total_seen_number_of_cycles = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, (cj_aug_cycle_curve_data, cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, weights, end_input_ids, end_attn_mask, dataset_ids, seen_unseen_ids, DKP_embeddings) in tqdm(enumerate(test_loader)):
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device) # [B, L, 3, fixed_length_of_curve]
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
            labels = labels.float().to(accelerator.device)
            input_ids = input_ids.int().to(accelerator.device)
            attention_mask = attention_mask.int().to(accelerator.device)
            dataset_ids = dataset_ids.int().to(accelerator.device)

            seen_number_of_cycles = torch.sum(curve_attn_mask, dim=1) # [B]


            outputs, _, _, _, _, _, _  = model(cycle_curve_data, curve_attn_mask, 
            input_ids=input_ids, attention_mask=attention_mask, end_input_ids=end_input_ids, end_attn_mask=end_attn_mask,
            DKP_embeddings=DKP_embeddings)
            # self.accelerator.wait_for_everyone()
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            sample_size += transformed_preds.shape[0]
            
            all_predictions, all_targets, dataset_ids, seen_unseen_ids, seen_number_of_cycles = accelerator.gather_for_metrics((transformed_preds, transformed_labels, dataset_ids, seen_unseen_ids, seen_number_of_cycles))

            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            total_dataset_ids = total_dataset_ids + dataset_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()
            total_seen_number_of_cycles = total_seen_number_of_cycles + seen_number_of_cycles.detach().cpu().numpy().reshape(-1).tolist()

    res_path='./results'
    # accelerator.wait_for_everyone()
    accelerator.set_trigger()
    if accelerator.check_trigger():
        total_dataset_ids = np.array(total_dataset_ids)
        total_references = np.array(total_references)
        total_seen_unseen_ids = np.array(total_seen_unseen_ids)
        total_seen_number_of_cycles = np.array(total_seen_number_of_cycles)
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

        if eval_cycle_min is None or eval_cycle_max is None:
            calculate_metrics_based_on_seen_number_of_cycles(total_preds, total_references, total_seen_number_of_cycles, alpha, alpha2, args.model, dataset, finetune_dataset=finetune_dataset, start=args.seq_len, end=args.early_cycle_threshold, seed=args.seed)
