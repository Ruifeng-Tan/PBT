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
from models import CPMLP, CPTransformer
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_evaluate
import time
import random
import numpy as np
import os
import json
import datetime
from data_provider.data_factory import data_provider_baseline
import joblib
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, vali_baseline
from layers.Adapters import CPLayerWithAdapter, tLayerWithAdapter, CPTtLayerWithAdapter

parser = argparse.ArgumentParser(description='Time-LLM')

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
parser.add_argument('--save_evaluation_res', action='store_true', default=False, help='the True to save the results; Only effective when eval_cycle_min or eval_cycle_max is smaller than 0')
parser.add_argument('--alpha1', type=float, default=0.15, help='the alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the alpha for alpha-accuracy')
parser.add_argument('--args_path', type=str, help='the path to the pretrained model parameters')

# finetune 
parser.add_argument('--finetune_dataset', type=str, help='the target dataset for model finetuning')
parser.add_argument('--finetune_method', type=str, default='FT', help='the fine-tuning method. [FT, EFT, AT]')
parser.add_argument('--adapter_size', type=int, default=32, help='the path to the pretrained model parameters')

args = parser.parse_args()
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2_baseline.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# load from the saved path
args_path = args.args_path
dataset = args.finetune_dataset
batch_size = args.batch_size
finetune_method = args.finetune_method
args_json = json.load(open(f'{args_path}args.json'))
trained_dataset = args_json['dataset']
finetune_dataset = args.finetune_dataset
args_json['dataset'] = dataset
args_json['batch_size'] = batch_size
args_json['alpha1'] = args.alpha1
args_json['alpha2'] = args.alpha2
args_json['save_path'] = args.checkpoints
args_json['model'] = args.model
args_json['finetune_method'] = args.finetune_method
args_json['adapter_size'] = args.adapter_size
args.__dict__ = args_json
for ii in range(args.itr):
    # setting record of experiments
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

    data_provider_func = data_provider_baseline
    if args.model == 'CPMLP':
        model = CPMLP.Model(args).float()
    elif args.model == 'CPTransformer':
        model = CPTransformer.Model(args).float()
    else:
        print('The model is not an implemented baseline!')
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #         'deepset/sentence_bert',
    #         # 'huggyllama/llama-7b',
    #         trust_remote_code=True
    #     ) # The ouput of the tokenizer won't be used. We just randomly assign a tokenizer here.
    # if tokenizer.eos_token:
    #     tokenizer.pad_token = tokenizer.eos_token
    # else:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    label_scaler = joblib.load(f'{args_path}/label_scaler')
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    life_class_scaler = joblib.load(f'{args_path}/life_class_scaler')

    path = os.path.join(args.save_path, setting + '-' + args.model_comment)  # unique checkpoint saving path
    accelerator.print(args.save_path)
    accelerator.print(path)

    train_data, train_loader = data_provider_func(args, 'train', None, sample_weighted=args.weighted_sampling)
    label_scaler = train_data.return_label_scaler()  
    life_class_scaler = train_data.return_life_class_scaler()      
    
    accelerator.print("Loading training samples......")
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)

    if accelerator.is_local_main_process and os.path.exists(path):
        del_files(path)  # delete checkpoint files
        accelerator.print(f'success delete {path}')

    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{path}/label_scaler')
    joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    time_now = time.time()

    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    criterion = nn.MSELoss(reduction='none') 
    accumulation_steps = args.accumulation_steps
    load_checkpoint_in_model(model, args_path) # load the saved parameters into model
    accelerator.print(f'The model is {args.model}')
    accelerator.print(f'load model from:\n {args_path}')
    accelerator.print(f'Model is loaded!')

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

    train_steps = len(train_loader)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(train_loader, vali_loader, test_loader, model, model_optim, scheduler)
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
        total_preds, total_references = [], []
        for i, (cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids) in enumerate(train_loader):
            with accelerator.accumulate(model):
                model_optim.zero_grad()
                iter_count += 1
                
                life_class = life_class.to(accelerator.device)
                scaled_life_class = scaled_life_class.float().to(accelerator.device)
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                labels = labels.float().to(accelerator.device)
                
                # encoder - decoder
                outputs = model(cycle_curve_data, curve_attn_mask)

                cut_off = labels.shape[0]
                    
                if args.loss == 'MSE':
                    loss = criterion(outputs[:cut_off], labels)
                    loss = torch.mean(loss * weights)
                elif args.loss == 'MAPE':
                    tmp_outputs = outputs[:cut_off] * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                    loss = torch.mean(loss * weights)
                    
                label_loss = loss.detach().float()
                
                print_loss = loss.detach().float()
                
                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss
                total_lc_loss += print_life_class_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
                
                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
                accelerator.backward(loss)
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | lc_loss: {print_life_class_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        train_rmse = root_mean_squared_error(total_references, total_preds)
        train_mape = mean_absolute_percentage_error(total_references, total_preds)
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False)
        test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_baseline(args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True)
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
        total_cl_loss = total_cl_loss / len(train_loader)
        total_lc_loss = total_lc_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train lc loss: {total_lc_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        
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
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Test 15%-accuracy: {best_test_alpha_acc1:.4f} | Test 10%-accuracy: {best_test_alpha_acc2:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f} | Val 15%-accuracy: {best_vali_alpha_acc1:.4f} | Val 10%-accuracy: {best_vali_alpha_acc2:.4f} ')
accelerator.print(f'Best model performance: Test Seen MAPE: {best_seen_test_MAPE:.4f} | Test Unseen MAPE: {best_unseen_test_MAPE:.4f}')
accelerator.print(f'Best model performance: Test Seen 15%-accuracy: {best_seen_test_alpha_acc1:.4f} | Test Unseen 15%-accuracy: {best_unseen_test_alpha_acc1:.4f}')
accelerator.print(f'Best model performance: Test Seen 10%-accuracy: {best_seen_test_alpha_acc2:.4f} | Test Unseen 10%-accuracy: {best_unseen_test_alpha_acc2:.4f}')
accelerator.print(path)
accelerator.set_trigger()