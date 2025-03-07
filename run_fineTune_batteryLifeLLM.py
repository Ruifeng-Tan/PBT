# curriculum learning
import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
from utils.tools import train_model_course, get_parameter_number
from utils.losses import Battery_life_CL_loss
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from models import Autoformer, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr, BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial, BatteryLifeLLMv12_SDCM, BatteryLifeLLMv13_SDCM, BatteryLifeLLMv18_Stack, BatteryLifeLLMv6, BatteryLifeLLMv7_FlattenHead, BatteryLifeLLMv7_LinearHead, BatteryLifeLLMv7_LinearHead2, BatteryLifeLLMv7_pe, BatteryLifeLLMv8, DLinear, TimeLLM, OnefitsAll, \
    BatteryLifeLLMv5_distilling_version,BatteryLifeLLMv5_simpleLSTM,\
        BatteryLifeLLMv6_redescribe, BatteryLifeLLMv7,BatteryLifeLLMv7_GRUHead_tuneLLM
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_LLMv2
import time
import json
import random
import numpy as np
import os
import json
import datetime
import joblib
def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_batteryLifeLLM, load_content
wandb.login(key="90b2c598dc4a58105fdcdd8c03ec271984ec7417")
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
parser.add_argument('--save_path', type=str, required=False, default='long_term_forecast',
                    help='the save path of pretrained BatteryLifeLLM weights')
parser.add_argument('--target_dataset', type=str, required=False, default='long_term_forecast',
                    help='the name of the target dataset')
parser.add_argument('--batch_size', type=int, required=False, default=4,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.00005,
                    help='learning rate')
parser.add_argument('--eval_cycle_min', type=int, default=10, help='The lower bound for evaluation')
parser.add_argument('--eval_cycle_max', type=int, default=10, help='The upper bound for evaluation')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
args = parser.parse_args()
eval_cycle_min = args.eval_cycle_min
eval_cycle_max = args.eval_cycle_max
train_epochs = args.train_epochs
if eval_cycle_min < 0 or eval_cycle_max <0:
    eval_cycle_min = None
    eval_cycle_max = None
args_path = args.save_path
dataset = args.target_dataset
batch_size = args.batch_size
learning_rate = args.learning_rate
patience = args.patience

args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args_json['learning_rate'] = learning_rate
args_json['train_epochs'] = train_epochs
args_json['patience'] = patience
args.__dict__ = args_json

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_dm{}_nh{}_el{}_dl{}_df{}_llmLayers{}_{}_{}_Lora{}_lstmlayers{}_lradj{}_cl{}_dataset{}_curriculum{}_loss{}_stride{}_pathLen{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.llm_layers,
        args.des, ii, args.use_LoRA, args.lstm_layers, args.lradj, args.use_contrastive_learning, trained_dataset, args.use_curriculum_learning, args.loss, args.stride, args.patch_len)


    data_provider_func = None
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'OFT':
        model = OnefitsAll.Model(args).float()
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv6.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv5_distilling_version':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv5_distilling_version.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv5_simpleLSTM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv5_simpleLSTM.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv6_redescribe':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv6_redescribe.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_pe_cd':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_FlattenHead.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_pe':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_pe.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv18_Stack.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv8_tuneNorm':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Trial.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_pe_fix':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv12_SDCM.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv8':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv8.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_GRUHead_tuneLLM':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_GRUHead_tuneLLM.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_CDPatch':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead2.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_decompose':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv13_SDCM.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv9_pe':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv10_TrialP2_noDKP_SDCM_Tr.Model(model_config)
        data_provider_func = data_provider_LLMv2
    elif args.model == 'BatteryLifeLLMv7_LinearHead':
        model_ec_config = BatteryElectrochemicalConfig(args.__dict__)
        model_text_config = AutoConfig.from_pretrained(args.LLM_path)
        model_config = BatteryLifeConfig(model_ec_config, model_text_config)
        model = BatteryLifeLLMv7_LinearHead.Model(model_config)
        data_provider_func = data_provider_LLMv2
    else:
        model = TimeLLM.Model(args).float()
        
    tokenizer = None
    if 'BatteryLifeLLM' in args.model:
        tokenizer = model.tokenizer
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    label_scaler = joblib.load(f'{path}/label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    life_class_scaler = joblib.load(f'{path}/life_class_scaler')
    train_data, train_loader = data_provider_func(args, 'train', tokenizer, label_scaler, life_class_scaler=life_class_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max)      
    
    accelerator.print("Loading training samples......")
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', tokenizer, label_scaler, life_class_scaler=life_class_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', tokenizer, label_scaler, life_class_scaler=life_class_scaler, eval_cycle_min=eval_cycle_min, eval_cycle_max=eval_cycle_max)
    
    # if accelerator.is_local_main_process and os.path.exists(path):
    #     del_files(path)  # delete checkpoint files
    #     accelerator.print(f'success delete {path}')

    
    # accelerator.wait_for_everyone()
    # joblib.dump(label_scaler, f'{path}/label_scaler')
    # joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
    # with open(path+'/args.json', 'w') as f:
    #     json.dump(args.__dict__, f)
    if accelerator.is_local_main_process:
        wandb.init(
        # set the wandb project where this run will be logged
        project="LLM_finetune",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )
    # args.content = load_content(args)

    # load LoRA
    # print the module name

    # for n, m in model.named_modules():
    #     accelerator.print(n, m)

    if args.use_LoRA:
        # LLM_lora_config = LoraConfig(
        #     r=args.LoRA_r,
        #     lora_alpha=args.LoRA_r,
        #     lora_dropout=args.LoRA_dropOut,
        #     target_modules=["language_model.layers.31.self_attn.q_proj", "language_model.layers.31.self_attn.k_proj", "language_model.layers.31.self_attn.v_proj", "language_model.layers.31.self_attn.o_proj"],
        #     use_rslora=True, # sqrt(r)
        #     bias="none",
        #     modules_to_save=['head1', 'cpl']
        # )
        LLM_lora_config = LoraConfig(
            r=args.LoRA_r,
            lora_alpha=args.LoRA_r,
            lora_dropout=args.LoRA_dropOut,
            target_modules=["self_attn.q_proj", "self_attn.v_proj"],
            use_rslora=True, # sqrt(r)
            bias="none",
            modules_to_save=['head1', 'cpl']
        )
        # model.add_adapter(LLM_lora_config)
        model = get_peft_model(model, LLM_lora_config)
        model.print_trainable_parameters()
    else:
        para_res = get_parameter_number(model)
        accelerator.print(para_res)
        
    for name, module in model._modules.items():
        accelerator.print(name," : ",module)
    
    
    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    trained_parameters_names = []
    for name, p in model.named_parameters():
        if 'head1' in name or 'cpl' in name:
            p.requires_grad = True
            trained_parameters_names.append(name)
            trained_parameters.append(p)
        else:
            p.requires_grad = False

    accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

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
    criterion = nn.MSELoss() 
    life_class_criterion = nn.MSELoss() 
    cl_criterion = Battery_life_CL_loss(class_numbers)

    load_checkpoint_in_model(model, path) # load the saved parameters into model
    path = path + '_f' # new save path
    os.makedirs(path, exist_ok=True) # make the saving path to save the finetuned model
    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{path}/label_scaler')
    joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)


    best_vali_loss = float('inf')
    best_vali_MAE, best_test_MAE = 0, 0
    best_vali_RMSE, best_test_RMSE = 0, 0
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
        for i, (cycle_curve_data, curve_attn_mask, input_ids, attention_mask, labels, life_class, scaled_life_class) in enumerate(train_loader):
            with accelerator.accumulate(model):
                # batch_x_mark is the total_masks
                # batch_y_mark is the total_used_cycles
                iter_count += 1
                
                life_class = life_class.to(accelerator.device)
                scaled_life_class = scaled_life_class.float().to(accelerator.device)
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                labels = labels.float().to(accelerator.device)
                input_ids = input_ids.int().to(accelerator.device)
                attention_mask = attention_mask.int().to(accelerator.device)
                
                

                # encoder - decoder
                outputs, features, preds_life_class = model(cycle_curve_data, curve_attn_mask, input_ids=input_ids, attention_mask=attention_mask, contrastive_learning=args.use_contrastive_learning)
                
                if args.use_contrastive_learning:
                    labels = torch.cat([labels, labels], dim=0)
                    scaled_life_class = torch.cat([scaled_life_class, scaled_life_class], dim=0)
                    cut_off = labels.shape[0]//2
                else:
                    cut_off = labels.shape[0]
                    
                if args.loss == 'MSE':
                    loss = criterion(outputs, labels)
                if args.loss == 'MAPE':
                    tmp_outputs = outputs * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                    
                label_loss = loss.detach().float()
                
                if args.use_contrastive_learning:
                    life_class_loss = life_class_criterion(preds_life_class, scaled_life_class)
                    cl_loss = cl_criterion(features, life_class)
                    print_cl_loss = cl_loss.detach().float()
                    print_life_class_loss = life_class_loss.detach().float()
                    loss = loss + args.cl_loss_weight * (cl_loss + life_class_loss)
                
                
                print_loss = loss.detach().float()
                
                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss
                total_lc_loss += print_life_class_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
                
                mae_metric.add_batch(
                    predictions = all_predictions,
                    references = all_targets
                )
                mape_metric.add_batch(
                    predictions = all_predictions,
                    references = all_targets
                )
                accelerator.backward(loss)
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                model_optim.zero_grad()
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | lc_loss: {print_life_class_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        train_mae_res = mae_metric.compute()
        train_mape_res = mape_metric.compute()
        train_mae = train_mae_res['mae']
        train_mape = train_mape_res['mape']
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_loss, vali_mae_loss, vali_mape = vali_batteryLifeLLM(args, accelerator, model, vali_data, vali_loader, criterion)
        test_loss, test_mae_loss, test_mape = vali_batteryLifeLLM(args, accelerator, model, test_data, test_loader, criterion)
        test_rmse, vali_rmse = np.sqrt(test_loss), np.sqrt(vali_loss)
        if args.loss == 'MAPE':
            vali_loss = vali_mape
        elif args.loss == 'MSE':
            pass

        
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
        total_lc_loss = total_lc_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train lc loss: {total_lc_loss:.5f} | Train MAE: {train_mae:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_MAE": vali_mae_loss, "vali_RMSE":vali_rmse, "test_MAE": test_mae_loss, "test_RMSE": test_rmse})
        
        early_stopping(vali_loss, vali_mae_loss, test_mae_loss, model, path)
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

accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f}')
accelerator.print(path)
accelerator.set_trigger()
if accelerator.check_trigger() and accelerator.is_local_main_process:
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_MAE": best_vali_MAE, "vali_RMSE": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "test_MAE": best_test_MAE, "test_RMSE": best_test_RMSE, "test_MAPE":best_test_MAPE})
    wandb.finish()
