from data_provider.data_loader import Dataset_BatteryLifeLLM_original, DomainBalancedBatchSampler
from data_provider.data_loader import my_collate_fn, my_collate_fn_withId
from torch.utils.data import DataLoader, RandomSampler, Dataset

data_dict = {
    'Dataset_BatteryLifeLLM_original': Dataset_BatteryLifeLLM_original
}

def data_provider_baseline_DA(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, temperature2mask=None, format2mask=None, cathodes2mask=None, anode2mask=None, target_dataset='None', meta_learning=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if flag == 'val' or flag == 'test':
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                temperature2mask=temperature2mask,
                format2mask=format2mask,
                cathodes2mask=cathodes2mask,
                anode2mask=anode2mask,
                use_target_dataset=True
            )
    else:
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                temperature2mask=temperature2mask,
                format2mask=format2mask,
                cathodes2mask=cathodes2mask,
                anode2mask=anode2mask
            )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn)
    
    if target_dataset != 'None' and flag=='train':
        target_data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=data_set.return_label_scaler(),
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                temperature2mask=temperature2mask,
                format2mask=format2mask,
                cathodes2mask=cathodes2mask,
                anode2mask=anode2mask,
                use_target_dataset=True
            )

        target_data_loader = DataLoader(
                    target_data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=my_collate_fn)
        target_sampler = RandomSampler(target_data_loader.dataset, replacement=True, num_samples=len(data_loader.dataset))
        target_resampled_dataloader = DataLoader(target_data_loader.dataset, batch_size=batch_size, sampler=target_sampler, collate_fn=my_collate_fn)
        return data_set, data_loader, target_data_set, target_resampled_dataloader
    else:
        return data_set, data_loader
    
def data_provider_LLMv2(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, temperature2mask=None, format2mask=None, cathodes2mask=None, anode2mask=None, meta_learning=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts, 
            total_charge_discharge_curves=total_charge_discharge_curves, 
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler,
            temperature2mask=temperature2mask,
            format2mask=format2mask,
            cathodes2mask=cathodes2mask,
            anode2mask=anode2mask
        )
    
    if meta_learning:
        sampler = DomainBalancedBatchSampler(data_set.total_domain_ids, batch_size, min_domains=2, shuffle=True)
        data_loader = DataLoader(
                    data_set,
                    num_workers=args.num_workers,
                    collate_fn=my_collate_fn, batch_sampler=sampler) # use the sampler
    else:
        data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=my_collate_fn)
        
    return data_set, data_loader


def data_provider_LLM_evaluate(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, temperature2mask=None, format2mask=None, cathodes2mask=None, anode2mask=None):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts, 
            total_charge_discharge_curves=total_charge_discharge_curves, 
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler,
            temperature2mask=temperature2mask,
            format2mask=format2mask,
            cathodes2mask=cathodes2mask,
            anode2mask=anode2mask
        )


    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_withId)
    return data_set, data_loader