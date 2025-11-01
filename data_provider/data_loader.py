import os
import random
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import copy
from torch.utils.data.sampler import Sampler, BatchSampler
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from Prompts.Mapping_helper import Mapping_helper
from utils.timefeatures import time_features
import warnings
import pickle
from sklearn.cluster import k_means
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
from torch.nn.utils.rnn import pad_sequence
from batteryml.data.battery_data import BatteryData
from utils.augmentation import BatchAugmentation_battery_revised
from data_provider.data_split_recorder import split_recorder
from data_provider.gate_masker import gate_masker
import accelerate
warnings.filterwarnings('ignore')
datasetName2ids = {
    'CALCE':0,
    'HNEI':1,
    'HUST':2,
    'MATR':3,
    'RWTH':4,
    'SNL':5,
    'MICH':6,
    'MICH_EXP':7,
    'Tongji1':8,
    'Stanford':9,
    'ISU-ILCC':11,
    'XJTU':12,
    'ZN-coin':13,
    'UL-PUR':14,
    'Tongji2':15,
    'Tongji3':16,
    'CALB':17,
    'ZN42':22,
    'ZN2024':23,
    'CALB42':24,
    'CALB2024':25,
    'NA-ion':27,
    'NA-ion42':28,
    'NA-ion2024':29,
}

import random
from torch.utils.data.sampler import Sampler
import numpy as np

class DomainBatchSampler(Sampler):
    def __init__(self, domain_ids, num_domains, batch_size, shuffle=True):
        self.domain_ids = domain_ids
        self.num_domains = num_domains
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert len(np.unique(domain_ids)) >= num_domains
        if self.batch_size % self.num_domains != 0:
            raise ValueError("batch_size must be divisible by num_domains")
        self.s_per_batch = self.batch_size // self.num_domains

        # Group indices by domain_id
        self.domain_to_indices_original = {}
        for idx, domain_id in enumerate(domain_ids):
            self.domain_to_indices_original.setdefault(domain_id, []).append(idx)
        
        self.domain_ids_unique = list(self.domain_to_indices_original.keys())
        self.total_samples = len(domain_ids)
        self.total_batches = self._compute_total_batches()

    def _compute_total_batches(self):
        # Estimate considering even distribution; actual batches may vary
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        domain_to_indices = {domain: idx.copy() for domain, idx in self.domain_to_indices_original.items()}
        fresh_domain_to_indices = domain_to_indices.copy()
        domain_list = self.domain_ids_unique.copy()

        if self.shuffle:
            random.shuffle(domain_list)
            for domain in domain_list:
                random.shuffle(domain_to_indices[domain])

        batch_count = 0
        while batch_count < self.total_batches:
            # Collect domains with at least one sample
            available_domains = [d for d in domain_list if len(domain_to_indices[d]) > 0]
            if not available_domains:
                break

            # Select domains, allowing repeats if necessary
            if len(available_domains) < self.num_domains:
                # The remaining domains are less than num_domains. We will randomly retrieve some exhausted domains to fill the gap.
                exhausted_domains = [d for d in domain_list if len(domain_to_indices[d]) == 0]
                selected_domains = available_domains + random.sample(exhausted_domains, k=self.num_domains-len(available_domains))
            else:
                selected_domains = random.sample(available_domains, self.num_domains)

            batch = []
            for domain in selected_domains:
                indices = self.get_sample_indices_from_domain(fresh_domain_to_indices, domain_to_indices, domain=domain)
                batch.extend(indices)

            # Yield even if batch is smaller than batch_size (due to s < s_per_batch)
            batch_count += 1
            yield batch

    def __len__(self):
        return self.total_batches
    
    def get_sample_indices_from_domain(self, fresh_domain_to_indices, domain_to_indices, domain):
        '''
        Get sample indices from a domain.
        If the domain is not exhuasted:
            If the domain can provide enough samples i.e. self.s_per_batch, we will get the samples from it and update the domain_to_indices for it.
            If the domain cannot provide enough samples i.e. self.s_per_batch, we will get the remaining samples and randomly get some samples from this domain. Then, mark the domain as exhausted.
        else:
            Randomly get sampels from this domain.
        Return:
            indices
        '''
        if len(domain_to_indices[domain]) > 0:
            # The domain is not exhausted
            if len(domain_to_indices[domain]) >= self.s_per_batch:
                indices = domain_to_indices[domain]
                retunred_indices = indices[:self.s_per_batch]
                domain_to_indices[domain] = indices[self.s_per_batch:]
            else:
                indices = domain_to_indices[domain]
                gap_sample_size = self.s_per_batch - len(indices)
                retunred_indices = indices + random.sample(fresh_domain_to_indices[domain], k=gap_sample_size)
                domain_to_indices[domain] = []
        else:
            # This domain is exhuasted
            retunred_indices = random.sample(fresh_domain_to_indices[domain], k=self.s_per_batch)
        return retunred_indices

def my_collate_fn_withId(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    # cj_aug_cycle_curve_data = torch.vstack([i['cj_cycle_curve_data'].unsqueeze(0) for i in samples])
    # fm_aug_cycle_curve_data = torch.vstack([i['fm_aug_cycle_curve_data'].unsqueeze(0) for i in samples])
    # m = torch.ones((fm_aug_cycle_curve_data.shape[0],1,1,1), dtype=fm_aug_cycle_curve_data.dtype, device=fm_aug_cycle_curve_data.device)
    # m = m.uniform_(0, 1) < 0.5 # set True to use cut_aug
    # m = m.expand_as(fm_aug_cycle_curve_data)

    # aug_cycle_curve_data = torch.where(m, cj_aug_cycle_curve_data, fm_aug_cycle_curve_data) # randomly use frequency mask and cutoff_jitter

    cathode_masks = torch.vstack([i['cathode_mask'] for i in samples])
    temperature_masks = torch.vstack([i['temperature_mask'] for i in samples])
    format_masks = torch.vstack([i['format_mask'] for i in samples])
    anode_masks = torch.vstack([i['anode_mask'] for i in samples])
    ion_type_masks = torch.vstack([i['ion_type_mask'] for i in samples])
    combined_masks = torch.vstack([i['combined_mask'] for i in samples])

    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])

    labels = torch.Tensor([i['labels'] for i in samples])

    weights = torch.Tensor([i['weight'] for i in samples])
    
    DKP_embeddings = torch.vstack([i['DKP_embedding'] for i in samples])
    dataset_ids = torch.Tensor([i['dataset_id'] for i in samples])
    domain_ids = torch.Tensor([i['domain_ids'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])

    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
    cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
    
    return cycle_curve_data, curve_attn_mask, labels, weights, dataset_ids, seen_unseen_ids, DKP_embeddings, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids

def my_collate_fn(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    # cj_aug_cycle_curve_data = torch.vstack([i['cj_cycle_curve_data'].unsqueeze(0) for i in samples])

    file_names = [i['file_name'] for i in samples]
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])

    labels = torch.Tensor([i['labels'] for i in samples])
    weights = torch.Tensor([i['weight'] for i in samples])

    cathode_masks = torch.vstack([i['cathode_mask'] for i in samples])
    temperature_masks = torch.vstack([i['temperature_mask'] for i in samples])
    format_masks = torch.vstack([i['format_mask'] for i in samples])
    anode_masks = torch.vstack([i['anode_mask'] for i in samples])
    ion_type_masks = torch.vstack([i['ion_type_mask'] for i in samples])
    combined_masks = torch.vstack([i['combined_mask'] for i in samples])


    DKP_embeddings = torch.vstack([i['DKP_embedding'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])
    domain_ids = torch.Tensor([i['domain_ids'] for i in samples])

    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
    cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

    return cycle_curve_data, curve_attn_mask, labels, weights, file_names, DKP_embeddings, seen_unseen_ids, cathode_masks, temperature_masks, format_masks, anode_masks, ion_type_masks, combined_masks, domain_ids

# BatterLifeLLM dataloader
class Dataset_PBT(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, tokenizer=None, eval_cycle_max=None, eval_cycle_min=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, temperature2mask=None, format2mask=None, cathodes2mask=None, 
                 anode2mask=None, ion2mask=None, trained_dataset=None, use_target_dataset=False):
        '''
        init the Dataset_BatteryFormer class
        :param args:model parameters
        :param flag:including train, val, test
        :param scaler:scaler or not
        '''
        self.llm_choice = args.llm_choice
        self.eval_cycle_max = eval_cycle_max
        self.eval_cycle_min = eval_cycle_min
        self.tokenizer = tokenizer
        self.args = args
        self.seed = args.seed
        self.root_path = args.root_path
        self.seq_len = args.seq_len
        self.charge_discharge_len = args.charge_discharge_length  # The resampled length for charge and discharge curves
        self.flag = flag
        self.dataset = args.dataset if not use_target_dataset else args.target_dataset
        self.early_cycle_threshold = args.early_cycle_threshold
        self.cathode_json = json.load(open('./gate_data/cathodes.json'))
        self.cathode_experts = args.cathode_experts
        self.temperature_json = json.load(open('./gate_data/temperatures.json'))
        self.temperature_experts = args.temperature_experts
        self.format_json = json.load(open('./gate_data/formats.json'))
        self.format_experts = args.format_experts
        self.anode_json = json.load(open('./gate_data/anodes.json'))
        self.anode_experts = args.anode_experts
        self.ion_experts = args.ion_experts
        self.trained_dataset = trained_dataset

        self.temperature2mask = temperature2mask
        self.format2mask = format2mask
        self.cathodes2mask = cathodes2mask
        self.anode2mask = anode2mask
        self.ion2mask = ion2mask

        self.name2domainID = json.load(open(f'/data/trf/python_works/BatteryMoE/gate_data/name2agingConditionID.json'))

        self.label_prompts_vectors = {}
        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
        self.aug_helper = BatchAugmentation_battery_revised()
        assert flag in ['train', 'test', 'val']
        if self.dataset == 'exp':
            self.train_files = split_recorder.HUST_train_files + split_recorder.Stanford_train_files
            self.val_files = split_recorder.HUST_val_files + split_recorder.Stanford_val_files
            self.test_files =  split_recorder.HUST_test_files + split_recorder.Stanford_test_files
        elif self.dataset == 'Tongji':
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == 'HUST':
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == 'MATR':
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == 'SNL':
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == 'MICH':
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == 'MICH_EXP':
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == 'UL_PUR':
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == 'RWTH':
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == 'HNEI':
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == 'CALCE':
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == 'Stanford':
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == 'ISU_ILCC':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'ISU_ILCC2':
            # ISU_ILCC_delG49C1
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_delG49C1_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'XJTU':
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == 'MIX_large':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'ZN-coin':
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files 
            self.test_files = split_recorder.ZNcoin_test_files   
        elif self.dataset == 'CALB':
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files 
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == 'ZN-coin42':
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == 'ZN-coin2024':
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == 'CALB42':
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == 'CALB2024':
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == 'NAion':
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == 'NAion42':
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == 'NAion2024':
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        elif self.dataset == 'MIX_all':
            self.train_files = split_recorder.MIX_all_train_files
            self.val_files = split_recorder.MIX_all_val_files
            self.test_files = split_recorder.MIX_all_test_files
        elif self.dataset == 'MIX_all42':
            self.train_files = split_recorder.MIX_all_42_train_files
            self.val_files = split_recorder.MIX_all_42_val_files
            self.test_files = split_recorder.MIX_all_42_test_files
        elif self.dataset == 'MIX_all2024':
            self.train_files = split_recorder.MIX_all_2024_train_files
            self.val_files = split_recorder.MIX_all_2024_val_files
            self.test_files = split_recorder.MIX_all_2024_test_files
        elif self.dataset == 'MIX_fig_cathode_LFP':
            self.train_files = split_recorder.MIX_large_cathode_LFP_train_files
            self.val_files = split_recorder.MIX_large_carhode_LFP_val_files
            self.test_files = split_recorder.MIX_large_cathode_LFP_test_files
        elif self.dataset == 'MIX_fig_cathode_NCM':
            self.train_files = split_recorder.MIX_large_cathode_NCM_train_files + split_recorder.MIX_large_cathode_NCM840610_train_files + split_recorder.MIX_large_cathode_NCM111_train_files + split_recorder.MIX_large_cathode_NCM523_train_files + split_recorder.MIX_large_cathode_NCM831107_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM_val_files + split_recorder.MIX_large_cathode_NCM840610_val_files + split_recorder.MIX_large_cathode_NCM111_val_files + split_recorder.MIX_large_cathode_NCM523_val_files + split_recorder.MIX_large_cathode_NCM831107_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM_test_files + split_recorder.MIX_large_cathode_NCM840610_test_files + split_recorder.MIX_large_cathode_NCM111_test_files + split_recorder.MIX_large_cathode_NCM523_test_files + split_recorder.MIX_large_cathode_NCM831107_test_files
        elif self.dataset == 'MIX_fig_cathode_NCA':
            self.train_files = split_recorder.MIX_large_cathode_NCA811405_train_files + split_recorder.MIX_large_cathode_NCA801505_train_files + split_recorder.MIX_large_cathode_NCA861103_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCA811405_val_files + split_recorder.MIX_large_cathode_NCA801505_val_files + split_recorder.MIX_large_cahtode_NCA861103_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCA811405_test_files + split_recorder.MIX_large_cathode_NCA801505_test_files + split_recorder.MIX_large_cathode_NCA861103_test_files
        elif self.dataset == 'MIX_fig_cathode_NCMNCA':
            self.train_files = split_recorder.MIX_large_cathode_NCM_NCA_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM_NCA_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM_NCA_test_files
        elif self.dataset == 'MIX_fig_cathode_NCMLCO':
            self.train_files = split_recorder.MIX_large_cathode_NCM422LCO_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM422LCO_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM422LCO_test_files
        elif self.dataset == 'MIX_fig_cathode_LCO':
            self.train_files = split_recorder.MIX_large_cathode_LCO_train_files
            self.val_files = split_recorder.MIX_large_cathode_LCO_val_files
            self.test_files = split_recorder.MIX_large_cathode_LCO_test_files
        elif self.dataset == 'MIX_fig_anode_graphite':
            self.train_files = split_recorder.MIX_large_anode_graphite_train_files + split_recorder.MIX_large_anode_carbon_train_files + split_recorder.MIX_large_anode_graphite_PVDF_train_files + split_recorder.MIX_large_anode_AG_train_files
            self.val_files = split_recorder.MIX_large_anode_graphite_val_files + split_recorder.MIX_large_anode_carbon_val_files + split_recorder.MIX_large_anode_graphite_PVDF_val_files + split_recorder.MIX_large_anode_AG_val_files
            self.test_files = split_recorder.MIX_large_anode_graphite_test_files + split_recorder.MIX_large_anode_carbon_test_files + split_recorder.MIX_large_anode_graphite_PVDF_test_files + split_recorder.MIX_large_anode_AG_test_files
        elif self.dataset == 'MIX_fig_anode_graphite_si':
            self.train_files = split_recorder.MIX_large_anode_graphite_si_train_files
            self.val_files = split_recorder.MIX_large_anode_graphite_si_val_files
            self.test_files = split_recorder.MIX_large_anode_graphite_si_test_files
        elif self.dataset == 'MIX_fig_format_prismatic':
            self.train_files = split_recorder.MIX_large_format_prismatic_train_files
            self.val_files = split_recorder.MIX_large_format_prismatic_val_files
            self.test_files = split_recorder.MIX_large_format_prismatic_test_files
        elif self.dataset == 'MIX_fig_format_18650':
            self.train_files = split_recorder.MIX_large_format_18650_train_files
            self.val_files = split_recorder.MIX_large_format_18650_val_files
            self.test_files = split_recorder.MIX_large_format_18650_test_files
        elif self.dataset == 'MIX_fig_format_pouch':
            self.train_files = split_recorder.MIX_large_format_pouch_train_files 
            self.val_files = split_recorder.MIX_large_format_pouch_val_files 
            self.test_files = split_recorder.MIX_large_format_pouch_test_files
        elif self.dataset == 'MIX_fig_format_502030_pouch':
            self.train_files = split_recorder.MIX_large_format_502030_pouch_train_files
            self.val_files = split_recorder.MIX_large_format_502030_pouch_val_files
            self.test_files = split_recorder.MIX_large_format_502030_pouch_test_files 
        elif self.dataset == 'MIX_fig_format_4090132_pouch':
            self.train_files = split_recorder.MIX_large_format_4090132_pouch_train_files
            self.val_files = split_recorder.MIX_large_format_4090132_pouch_val_files
            self.test_files = split_recorder.MIX_large_format_4090132_pouch_test_files
        elif self.dataset == 'MIX_fig_temp_neg5':
            self.train_files = split_recorder.MIX_large_temp_5_train_files
            self.val_files = split_recorder.MIX_large_temp_5_val_files
            self.test_files = split_recorder.MIX_large_temp_5_test_files
        elif self.dataset == 'MIX_fig_temp_15':
            self.train_files = split_recorder.MIX_large_temp_15_train_files
            self.val_files = split_recorder.MIX_large_temp_15_val_files
            self.test_files = split_recorder.MIX_large_temp_15_test_files
        elif self.dataset == 'MIX_fig_temp_20':
            self.train_files = split_recorder.MIX_large_temp_20_train_files
            self.val_files = split_recorder.MIX_large_temp_20_val_files
            self.test_files = split_recorder.MIX_large_temp_20_test_files
        elif self.dataset == 'MIX_fig_temp_23':
            self.train_files = split_recorder.MIX_large_temp_23_train_files
            self.val_files = split_recorder.MIX_large_temp_23_val_files
            self.test_files = split_recorder.MIX_large_temp_23_test_files
        elif self.dataset == 'MIX_fig_temp_25':
            self.train_files = split_recorder.MIX_large_temp_25_train_files
            self.val_files = split_recorder.MIX_large_temp_25_val_files
            self.test_files = split_recorder.MIX_large_temp_25_test_files
        elif self.dataset == 'MIX_fig_temp_30':
            self.train_files = split_recorder.MIX_large_temp_30_train_files
            self.val_files = split_recorder.MIX_large_temp_30_val_files
            self.test_files = split_recorder.MIX_large_temp_30_test_files
        elif self.dataset == 'MIX_fig_temp_35':
            self.train_files = split_recorder.MIX_large_temp_35_train_files
            self.val_files = split_recorder.MIX_large_temp_35_val_files
            self.test_files = split_recorder.MIX_large_temp_35_test_files
        elif self.dataset == 'MIX_fig_temp_45':
            self.train_files = split_recorder.MIX_large_temp_45_train_files
            self.val_files = split_recorder.MIX_large_temp_45_val_files
            self.test_files = split_recorder.MIX_large_temp_45_test_files
        elif self.dataset == 'MIX_eval':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files
            self.test_files = split_recorder.MIX_large_val_files
        elif self.dataset == 'ISU_ILCC_eval_delG49C1':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_delG49C1_files
            self.test_files = split_recorder.ISU_ILCC_val_delG49C1_files
        elif self.dataset == 'MIX_large_ablation_75p' or self.dataset == 'MIX_75p':
            if self.args.seed == 42:
                self.train_files = split_recorder.MIX_large_reduced_train_75p_files_42
            elif self.args.seed == 2021:
                self.train_files = split_recorder.MIX_large_reduced_train_75p_files_2021
            elif self.args.seed == 2024:
                self.train_files = split_recorder.MIX_large_reduced_train_75p_files_2024
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_50p' or self.dataset == 'MIX_50p':
            if self.args.seed == 42:
                self.train_files = split_recorder.MIX_large_reduced_train_50p_files_42
            elif self.args.seed == 2021:
                self.train_files = split_recorder.MIX_large_reduced_train_50p_files_2021
            elif self.args.seed == 2024:
                self.train_files = split_recorder.MIX_large_reduced_train_50p_files_2024
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_25p' or self.dataset == 'MIX_25p':
            if self.args.seed == 42:
                self.train_files = split_recorder.MIX_large_reduced_train_25p_files_42
            elif self.args.seed == 2024:
                self.train_files = split_recorder.MIX_large_reduced_train_25p_files_2024
            elif self.args.seed == 2021:
                self.train_files = split_recorder.MIX_large_reduced_train_25p_files_2021
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'Stanford_formation' and self.args.seed == 2021:
            self.train_files = split_recorder.Stanford_formation_45_train_files_2021
            self.val_files = split_recorder.Stanford_formation_45_val_files_2021
            self.test_files = split_recorder.Stanford_formation_45_test_files_2021
        elif self.dataset == 'Stanford_formation' and self.args.seed == 42:
            self.train_files = split_recorder.Stanford_formation_45_train_files_42
            self.val_files = split_recorder.Stanford_formation_45_val_files_42
            self.test_files = split_recorder.Stanford_formation_45_test_files_42
        elif self.dataset == 'Stanford_formation' and self.args.seed == 2024:
            self.train_files = split_recorder.Stanford_formation_45_train_files_2024
            self.val_files = split_recorder.Stanford_formation_45_val_files_2024
            self.test_files = split_recorder.Stanford_formation_45_test_files_2024
        else:
            raise Exception(f'{self.dataset} is not supported!')

         
        # load the prompt embedding
        # The domain-knowledge prompt embeddings are only affected by the LLM and prompt
        train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all_{self.llm_choice}.pkl', 'rb'))
        val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all_{self.llm_choice}.pkl', 'rb'))
        test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all_{self.llm_choice}.pkl', 'rb'))

        # Stanford_formation_prompt_embeddings = pickle.load(open(f'{self.root_path}/training_DKP_embed_all_Llama_Stanford_formation.pkl', 'rb'))

        if self.dataset != 'Stanford_formation':
            self.cellName_prompt = train_part | val_part | test_part
        else:
            # self.cellName_prompt = Stanford_formation_prompt_embeddings
            pass

        if flag == 'train':
            self.files = [i for i in self.train_files]
        elif flag == 'val':
            self.files = [i for i in self.val_files]
        elif flag == 'test':

            self.files = [i for i in self.test_files]
            if self.seed == 2021:
                self.li_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test.json')) # this contains the 2021 records for Li, Zn and CALB
                self.na_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2021.json'))
                self.unseen_seen_record = self.li_ion_unseen_seen_record | self.na_ion_unseen_seen_record
            elif self.seed == 2024:
                self.li_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test.json'))
                self.na_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2024.json'))
                self.zn_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN2024.json'))
                self.calb_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB2024.json'))
                self.unseen_seen_record = self.li_ion_unseen_seen_record | self.na_ion_unseen_seen_record | self.zn_ion_unseen_seen_record | self.calb_unseen_seen_record
            elif self.seed == 42:
                self.li_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test.json'))
                self.na_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA42.json'))
                self.zn_ion_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN42.json'))
                self.calb_unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB42.json'))
                self.unseen_seen_record = self.li_ion_unseen_seen_record | self.na_ion_unseen_seen_record | self.zn_ion_unseen_seen_record | self.calb_unseen_seen_record

        self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.unique_labels, self.total_dataset_ids, self.total_center_vector_indices, self.total_file_names, self.total_cluster_labels, self.total_DKP_embeddings, self.total_seen_unseen_IDs, self.total_cathode_expert_masks, self.total_temperature_experts_masks, self.total_format_expert_masks, self.total_anode_expert_masks, self.total_ion_type_masks, self.total_combined_expert_masks, self.total_domain_ids = self.read_data()
        self.weights = self.get_loss_weight()
        if np.any(np.isnan(self.total_charge_discharge_curves)):
            raise Exception('Nan in the data')
        if np.any(np.isnan(self.unique_labels)):
            raise Exception('Nan in the labels')
        

        if flag == 'train' and label_scaler is None:
            self.label_scaler = StandardScaler()
            self.label_scaler.fit(np.array(self.unique_labels).reshape(-1, 1))
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1, 1))
        else:
            # validation set or testing set
            assert label_scaler is not None
            self.label_scaler = label_scaler
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1,1))


    def get_loss_weight(self, method='1/n'):
        '''
        Get the weight for weighted loss
        method can be ['1/n', '1/log(x+1)']
        '''
        if self.args.weighted_loss:
            if method == '1/n':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                weights = 1.0 / label_to_count[df["label"]].values
            elif method == '1/log(x+1)':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x)+1)
                weights = 1 / normalized_x
            else:
                raise Exception('Not implemented')
            return weights
        else:
            return np.ones(len(self.total_charge_discharge_curves))
    
    def get_center_vector_index(self, file_name):
        prefix = file_name.split('_')[0]
        if prefix in ['MATR', 'HUST'] or 'LFP' in file_name:
            return 0
        else:
            return 1 
        
    def return_label_scaler(self):
        return self.label_scaler
    
    
    def __len__(self):
        return len(self.total_labels)
        
    def read_data(self):
        '''
        read all data from files
        :return: x_enc, x_cycle_numbers, prompts, charge_data, discharge_data, RPT_masks, labels
        '''
    
        total_domain_ids = []
        total_charge_discharge_curves = []
        total_curve_attn_masks = []
        total_labels = [] # RUL
        unique_labels = []
        total_dataset_ids = []
        total_center_vector_indices = []
        total_file_names = []
        total_seen_unseen_IDs = []
        total_cathode_expert_masks = []
        total_temperature_experts_masks = []
        total_format_expert_masks = []
        total_anode_expert_masks = []
        total_ion_type_masks = []
        total_combined_expert_masks = []


        total_DKP_embeddings = []
        total_cluster_labels = []

        for file_name in tqdm(self.files):
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_id = datasetName2ids[file_name.split('_')[0]]
            else:
                dataset_id = datasetName2ids['MICH_EXP']


            # center_vector_index = self.get_center_vector_index(file_name)

            charge_discharge_curves, attn_masks, labels, eol = self.read_samples_from_one_cell(
                file_name)

            if eol is None:
                # This battery has not reached end of life
                continue

            if file_name in self.cathode_json:
                cathodes = self.cathode_json[file_name]
                cathodes = '_'.join(cathodes)
                cathode_mask = np.zeros(self.cathode_experts) # 1 indicates activated
                if cathodes in self.cathodes2mask:
                    cathode_index = self.cathodes2mask[cathodes]
                    cathode_mask[cathode_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the cathodes.json. We suggest the user to set the cathode in the cathodes.json and manually assign the expert'
                'using the cathodes2mask based on domain knowledge. When it is not possible to know the cathode or to manually assign the cathode, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                # cathode_mask = np.ones(self.cathode_experts) # assign according to the learned parameters
                # cathode_mask = np.zeros(self.cathode_experts) # only use the general experts

            cathode_mask = list(cathode_mask)

            if file_name in self.temperature_json:
                temperatures = self.temperature_json[file_name]
                temperature_mask = np.zeros(self.temperature_experts)
                if temperatures in self.temperature2mask:
                    temperature_index = self.temperature2mask[temperatures]
                    temperature_mask[temperature_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the temperatures.json. We suggest the user to set the temperature in the temperatures.json and manually assign the expert'
                'using the temperature2mask based on domain knowledge. When it is not possible to know the temperature or to manually assign the temperature, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                # temperature_mask = np.ones(self.temperature_experts) # assign according to the learned parameters
                # temperature_mask = np.zeros(self.temperature_experts) # only use the general experts

            temperature_mask = list(temperature_mask)

            if file_name in self.format_json:
                # format = '_'.join(self.format_json[file_name])
                format = self.format_json[file_name][0]
                format_mask = np.zeros(self.format_experts)
                if format in self.format2mask:
                    format_index = self.format2mask[format] 
                    format_mask[format_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the formats.json. We suggest the user to set the format in the formats.json and manually assign the expert'
                'using the format2mask based on domain knowledge. When it is not possible to know the format or to manually assign the format, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                # format_mask = np.ones(self.format_experts) # assign according to the learned parameters
                # format_mask = np.zeros(self.format_experts) # only use the general experts
            format_mask = list(format_mask)

            if file_name in self.anode_json:
                anode = self.anode_json[file_name][0]
                if anode == 'graphite' or anode == 'artificial graphite' or anode == 'carbon':
                    anode = 'graphite' # we assume other anodes are graphite
                anode_mask = np.zeros(self.anode_experts)
                if anode in self.anode2mask:
                    # if the anode is known, we assign the expert according to the domain knowledge
                    anode_index = self.anode2mask[anode]
                    anode_mask[anode_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the formats.json. We suggest the user to set the format in the formats.json and manually assign the expert'
                'using the format2mask based on domain knowledge. When it is not possible to know the format or to manually assign the format, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                # anode_mask = np.ones(self.anode_experts) # assign according to the learned parameters
                # anode_mask = np.zeros(self.anode_experts) # only use the general experts
            anode_mask = list(anode_mask)

            if self.ion_experts > 0:
                if self.flag == 'train':
                    if self.dataset.startswith('MIX_all'):
                        if file_name.startswith('ZN-coin'):
                            ion_index = self.ion2mask['Zn']
                            ion_type_mask = np.zeros(self.ion_experts)
                            ion_type_mask[ion_index] = 1
                        elif file_name.startswith('NA-ion'):
                            ion_index = self.ion2mask['Na']
                            ion_type_mask = np.zeros(self.ion_experts)
                            ion_type_mask[ion_index] = 1
                        else:
                            ion_index = self.ion2mask['Li']
                            ion_type_mask = np.zeros(self.ion_experts)
                            ion_type_mask[ion_index] = 1
                    else:
                        ion_type_mask = [] # ion experts are used only when many ion types are available in the training data
                else:
                    if self.trained_dataset is not None:
                        if self.trained_dataset.startswith('MIX_all'):
                            if file_name.startswith('ZN-coin'):
                                ion_index = self.ion2mask['Zn']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                            elif file_name.startswith('NA-ion'):
                                ion_index = self.ion2mask['Na']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                            else:
                                ion_index = self.ion2mask['Li']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                        else:
                            ion_type_mask = [] # ion experts are used only when many ion types are available in the training data
                    else:
                        if self.dataset.startswith('MIX_all'):
                            if file_name.startswith('ZN-coin'):
                                ion_index = self.ion2mask['Zn']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                            elif file_name.startswith('NA-ion'):
                                ion_index = self.ion2mask['Na']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                            else:
                                ion_index = self.ion2mask['Li']
                                ion_type_mask = np.zeros(self.ion_experts)
                                ion_type_mask[ion_index] = 1
                        else:
                            ion_type_mask = [] # ion experts are used only when many ion types are available in the training data
            else:
                ion_type_mask = []
            ion_type_mask = list(ion_type_mask)
            
            combined_expert_mask = cathode_mask + anode_mask + format_mask  + temperature_mask 

            cell_name = file_name.split('.pkl')[0]
            if self.flag == 'train':
                cluster_label = -1 # not used. Should be removed
            else:
                cluster_label = -1 # The cluster labels of validation or testing samples are unknown
            DKP_embedding = self.cellName_prompt[cell_name]
            domain_id = self.name2domainID[file_name]


            total_charge_discharge_curves += charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels 
            total_domain_ids += [domain_id for _ in range(len(labels))]
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            total_file_names += [file_name for _ in range(len(labels))]
            total_cluster_labels += [cluster_label for _ in range(len(labels))]
            total_DKP_embeddings += [DKP_embedding for _ in range(len(labels))]
            total_cathode_expert_masks += [cathode_mask for _ in range(len(labels))]
            total_format_expert_masks += [format_mask for _ in range(len(labels))]
            total_temperature_experts_masks += [temperature_mask for _ in range(len(labels))]
            total_anode_expert_masks += [anode_mask for _ in range(len(labels))]
            total_ion_type_masks += [ion_type_mask for _ in range(len(labels))]
            total_combined_expert_masks += [combined_expert_mask for _ in range(len(labels))]
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)
            if self.flag == 'test' and self.dataset != 'MIX_eval' and self.dataset != 'ISU_ILCC_eval_delG49C1':
                if self.dataset == 'Stanford_formation':
                    total_seen_unseen_IDs  += [0 for _ in range(len(labels))] # all formation protocols in the testing set are unseen
                else:
                    seen_unseen_id = self.unseen_seen_record[file_name]
                    if seen_unseen_id == 'unseen':
                        total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                    elif seen_unseen_id == 'seen':
                        total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                    else:
                        raise Exception('Check the bug!')
            else:
                total_seen_unseen_IDs += [1 for _ in range(len(labels))] # 1 indicates seen. This is not used on training or evaluation set

        return total_charge_discharge_curves, total_curve_attn_masks, np.array(total_labels), unique_labels, total_dataset_ids, total_center_vector_indices, total_file_names, total_cluster_labels, total_DKP_embeddings, total_seen_unseen_IDs, total_cathode_expert_masks, total_temperature_experts_masks, total_format_expert_masks, total_anode_expert_masks, total_ion_type_masks, total_combined_expert_masks, total_domain_ids
    
    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        prefix = file_name.split('_')[0]

        if prefix.startswith('MATR'):
            data =  pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data =  pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data =  pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data =  pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data =  pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            data =  pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data =  pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data =  pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))  
        elif prefix.startswith('UL-PUR'):
            data =  pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))  
        elif prefix.startswith('SMICH'):
            data =  pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb')) 
        elif prefix.startswith('BIT2'):
            data =  pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb')) 
        elif prefix.startswith('Tongji'):
            data =  pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            if self.dataset == 'Stanford_formation':
                data =  pickle.load(open(f'{self.root_path}/Stanford_formation/{file_name}', 'rb')) # read the formation data
            else:
                data =  pickle.load(open(f'{self.root_path}/Stanford/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data =  pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data =  pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data =  pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('NA-coin'):
            data =  pickle.load(open(f'{self.root_path}/NA-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data =  pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data =  pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))

        
        if prefix == 'MICH':
            with open(f'{self.root_path}/Life labels/total_MICH_labels.json') as f:
                life_labels = json.load(f)
        elif prefix.startswith('Tongji'):
            file_name = file_name.replace('--', '-#')
            with open(f'{self.root_path}/Life labels/Tongji_labels.json') as f:
                life_labels = json.load(f)
        elif prefix.startswith('Stanford'):
            if self.dataset == 'Stanford_formation':
                with open(f'{self.root_path}/Life labels/Stanford_2_labels.json') as f:
                    life_labels = json.load(f)
            else:
                with open(f'{self.root_path}/Life labels/{prefix}_labels.json') as f:
                    life_labels = json.load(f)
        else:
            with open(f'{self.root_path}/Life labels/{prefix}_labels.json') as f:
                life_labels = json.load(f)

        # file_name = file_name if self.dataset != 'Stanford_formation' else file_name.replace('Stanford_Formation_Nova_Formation-', 'Stanford_Nova_Regular_')
        if file_name in life_labels:
            eol = life_labels[file_name]
        else:
            eol = None
        print(file_name, eol)
        return data, eol
    
    def read_cell_df(self, file_name):
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, basic_prompt, eol
        '''
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]
        basic_prompt = self.generate_basic_prompt(cell_name)
    

        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
        SOC_interval = data['SOC_interval'] # get the charge and discharge soc interval
        SOC_interval = SOC_interval[1] - SOC_interval[0]
        cycle_data = data['cycle_data'] # list of cycle data dict
            
        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah']<0] = np.nan # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah']<0] = np.nan
            cycle_df.bfill(inplace=True) # deal with NaN
            total_cycle_dfs.append(cycle_df)
            
            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break
            
        df = pd.concat(total_cycle_dfs)
        # obtain the charge and discahrge curves
        if self.dataset == 'Stanford_formation':
            # nominal_capacity = 0.24
            charge_discharge_curves = self.get_charge_discharge_curves_Stanford_formation(file_name, df, self.early_cycle_threshold, nominal_capacity)
        else:
            charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold, nominal_capacity)
        return df, charge_discharge_curves, basic_prompt, eol, SOC_interval, nominal_capacity
      
    def generate_basic_prompt(self, cell_name):
        '''
        Generate the basic prompt that describes battery specifications and working conditions
        '''
        if 'CALB' in cell_name:
            bg_prompt = (
                        f"Task description: " 
                        f"The target is the number of cycles until the battery's discharge capacity reaches 90% of its nominal capacity. "
                        f"The discharge capacity is calculated under the described operating condition. "
                        f"Please directly output the target of the battery based on the provided data. "
                        )
        else:
            bg_prompt = (
                        f"Task description: " 
                        f"The target is the number of cycles until the battery's discharge capacity reaches 80% of its nominal capacity. "
                        f"The discharge capacity is calculated under the described operating condition. "
                        f"Please directly output the target of the battery based on the provided data. "
                        )
        
        # cell_name = cell_name if self.dataset != 'Stanford_formation' else cell_name.replace('Stanford_Formation_Nova_Formation-', 'Stanford_Nova_Regular_')
        helper = Mapping_helper(prompt_type='PROTOCOL', cell_name=cell_name)
        prompt = helper.do_mapping()
        if self.args.wo_DKPrompt:
            prompt = bg_prompt # remove the domain knowledge prompt
        else:
            prompt = bg_prompt + prompt
        return prompt
        
    def read_samples_from_one_cell(self, file_name):
        '''
        read all samples using this function
        :param file_name: which file needs to be read
        :return: history_sohs, future_sohs, masks, cycles, prompts, charge_data, discharge_data and RPT_masks in each sample
        '''

        df, charge_discharge_curves_data, basic_prompt, eol, SOC_interval, nominal_capacity = self.read_cell_df(file_name)
        if df is None or eol<=self.early_cycle_threshold:
            return None, None, None, None

        # the charge and discharge data
        charge_discharge_curves = []  # [N, seq_len, fix_charge_resample_len]
        attn_masks = []
        labels = []
        # get the early-life data
        early_charge_discharge_curves_data = charge_discharge_curves_data[:self.early_cycle_threshold]
        if np.any(np.isnan(early_charge_discharge_curves_data)):
            raise Exception(f'Failure in {file_name} | Early data contains NaN! Cycle life is {eol}!')
        for i in range(self.seq_len, self.early_cycle_threshold+1):
            if i >= eol:
                # If we encounter a battery whose cycle life is even smaller than early_cycle_threhold
                # We should not include the eol cycle data
                break
            
            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1 # set 1 not to mask
            if self.dataset == 'Stanford_formation':
                # tmp_attn_mask[7:] = 0 # mask the data after formation cycles
                padding_mask = (np.any(early_charge_discharge_curves_data != 0, axis=(-2, -1))).astype(int).reshape(-1) # mask the all zero curves, 0 indicates masked
                tmp_attn_mask = np.where(padding_mask==0, np.zeros_like(tmp_attn_mask), tmp_attn_mask)

            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if i >= self.eval_cycle_min and i <= self.eval_cycle_max:
                    # Only keep the val and test samples that satisfy the eval_cycle
                    pass
                else:
                    continue
            
            # tmp_cycle_data[i:] = np.zeros_like(tmp_cycle_data[i:])

            if self.args.wo_DKPrompt:
                tmp_prompt = basic_prompt
            else:
                tmp_prompt = basic_prompt
                # tmp_prompt = basic_prompt + f' Usage information: The battery has operated for {cycle_number} cycles. The current state of health is {last_soh}. '
            
            # if 'Instruct' in self.args.LLM_path:
            #     # Llama-instruct
            #     messages = [
            #         {"role": "system", "content": "You are an expert in predicting battery cycle life."},
            #         {"role": "user", "content": tmp_prompt}
            #     ]

            #     tmp_prompt = self.tokenizer.apply_chat_template(
            #         messages,
            #         tokenize=False,
            #         add_generation_prompt=True
            #     )
            # else:
            #     tmp_prompt = '<|begin_of_text|>' + tmp_prompt

            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            attn_masks.append(tmp_attn_mask)

        return charge_discharge_curves, attn_masks, labels, eol

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []

        prefix = file_name.split('_')[0]
        for cycle in range(1, early_cycle_threshold+1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]
                
                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records/nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values

                time_in_s_records = cycle_df['time_in_s'].values
                cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C<=-0.01) 
                discharge_end_index = cutoff_voltage_indices[0][-1]
                
                if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_35', 'CALB_45']:
                    # Every cycle first discharge and then charge
                    #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]
                    
                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity
                    
                    charge_voltages = charge_voltages[np.abs(charge_current_in_C)>0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C)>0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C)>0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C)>0.01]
                else:
                    # Every cycle first charge and then discharge
                    #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity
                    
                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C)>0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C)>0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C)>0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C)>0.01]
                    
                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(charge_voltages, charge_currents, charge_capacities)


                # if prefix in ['RWTH', 'OX']:
                #     voltage_records = np.concatenate([discharge_voltages, charge_voltages], axis=0)
                #     current_records = np.concatenate([discharge_currents, charge_currents], axis=0)
                #     capacity_in_battery = np.concatenate([discharge_capacities, charge_capacities], axis=0)
                # else:
                #     voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                #     current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                #     capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)
                
                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)
                
                # voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(voltage_records) # normalize using the cutoff voltage
                # current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                # capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the capacity
                voltage_records = voltage_records.reshape(1, self.charge_discharge_len)
                current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len)
                
                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))
              
        curves = np.concatenate(curves, axis=0) # [L, 3, fixed_len]
        return curves

    def get_charge_discharge_curves_Stanford_formation(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Designed for Stanford formation dataset
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []

        prefix = file_name.split('_')[0]
        for cycle in range(1, early_cycle_threshold+1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]
                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records/nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values

                cutoff_voltage_indices = np.nonzero(current_records_in_C>=-0.02) 
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                discharge_voltages = voltage_records[charge_end_index:]
                discharge_capacities = discharge_capacity_records[charge_end_index:] # The discharge capacity originally indicates the remaining capacity in the battery and hence should be adjusted.
                discharge_currents = current_records[charge_end_index:]
                discharge_capacities = max(discharge_capacities) - discharge_capacities # adjust the discharge capacity to indicate the discharged capacity

                charge_voltages = voltage_records[:charge_end_index]
                charge_capacities = charge_capacity_records[:charge_end_index]
                charge_currents = current_records[:charge_end_index]

                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0) # The formation data might contain no charge/discharge. To avoid errors, we resample the profile as a whole.
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)


                voltage_records, current_records, capacity_in_battery = self.resample_charge_discharge_curves(voltage_records, current_records, capacity_in_battery, self.charge_discharge_len)

                
                # voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                # current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                # capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)
                
               
                voltage_records = voltage_records.reshape(1, self.charge_discharge_len)
                current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len)
                
                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))
              
        curves = np.concatenate(curves, axis=0) # [L, 3, fixed_len]
        return curves
    
    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery, new_length=None):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or dicharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interploted records
        '''
        if new_length is not None:
            charge_discharge_len = new_length
        else:
            charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages)+1)
        interp_bases = np.linspace(1, len(voltages)+1, num=charge_discharge_len,
                                        endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery

    def __getitem__(self, index):
        # if 'Instruct' in self.args.LLM_path:
        #     # Llama
        #     end_of_the_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        #     #end_of_the_prompt = 'Predict battery cycle life'
        #     max_length = 5
        #     end_cut_off = - (max_length-1) # tokenizer will add begin_of_text, we don't need it in the end of prompt
        #     res = self.tokenizer(end_of_the_prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        # else:
        #     end_of_the_prompt = '<|end_of_text|>'
        #     max_length = 2
        #     end_cut_off = - (max_length-1)
        #     res = self.tokenizer(end_of_the_prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        # end_input_ids, end_attn_mask = res['input_ids'][0][end_cut_off:], res['attention_mask'][0][end_cut_off:]

        # prompt = self.total_prompts[index]
        # if self.args.wo_DKPrompt:
        #     max_length = 120
        #     end_cut_off = - (max_length-1) # we have already add the begin_of_text in the prompt
        #     res = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
        # else:
        #     max_length = 401
        #     end_cut_off = - (max_length-1) 
        #     res = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
        # input_ids, attention_mask = res['input_ids'][0][end_cut_off:], res['attention_mask'][0][end_cut_off:]
        # generate label prompt
        # max_length = 70
        # end_cut_off = - (max_length-1) 
        # label_prompt = self.total_label_prompts[index]
        # res = self.tokenizer(label_prompt, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
        # label_input_ids, label_attention_mask = res['input_ids'][0][end_cut_off:], res['attention_mask'][0][end_cut_off:]

        sample = {
                'cycle_curve_data': torch.Tensor(self.total_charge_discharge_curves[index]),
                'curve_attn_mask': torch.Tensor(self.total_curve_attn_masks[index]),
                'labels': self.total_labels[index],
                'weight': self.weights[index],
                'dataset_id': self.total_dataset_ids[index],
                'cathode_mask': torch.Tensor(self.total_cathode_expert_masks[index]),
                'anode_mask': torch.Tensor(self.total_anode_expert_masks[index]),
                'temperature_mask': torch.Tensor(self.total_temperature_experts_masks[index]),
                'format_mask': torch.Tensor(self.total_format_expert_masks[index]),
                'ion_type_mask': torch.Tensor(self.total_ion_type_masks[index]),
                'combined_mask': torch.Tensor(self.total_combined_expert_masks[index]),
                'DKP_embedding': torch.from_numpy(self.total_DKP_embeddings[index]),
                'cluster_label': self.total_cluster_labels[index],
                'file_name': self.total_file_names[index],
                'seen_unseen_id': self.total_seen_unseen_IDs[index],
                'domain_ids': self.total_domain_ids[index]
            }
        return sample
 

def my_collate_fn_withId_BL(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])
    # input_ids = pad_sequence([i['input_ids'] for i in samples], batch_first=True, padding_value=2)
    # attention_mask = pad_sequence([i['attention_mask'] for i in samples], batch_first=True, padding_value=0)
    life_class = torch.Tensor([i['life_class'] for i in samples])
    labels = torch.Tensor([i['labels'] for i in samples])
    scaled_life_class = torch.Tensor([i['scaled_life_class'] for i in samples])
    weights = torch.Tensor([i['weight'] for i in samples])
    dataset_ids = torch.Tensor([i['dataset_id'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])
    domain_ids = torch.Tensor([i['domain_ids'] for i in samples])
    return cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, dataset_ids, seen_unseen_ids, domain_ids

def my_collate_fn_baseline_BL(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])
    life_class = torch.Tensor([i['life_class'] for i in samples])
    labels = torch.Tensor([i['labels'] for i in samples])
    scaled_life_class = torch.Tensor([i['scaled_life_class'] for i in samples])
    weights = torch.Tensor([i['weight'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])
    domain_ids = torch.Tensor([i['domain_ids'] for i in samples])
    return cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids, domain_ids

# BatterLife dataloader
class Dataset_BatteryLife(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, tokenizer=None, eval_cycle_max=None, eval_cycle_min=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, use_target_dataset=False):
        '''
        init the Dataset_BatteryFormer class
        :param args:model parameters
        :param flag:including train, val, test
        :param scaler:scaler or not
        '''
        self.eval_cycle_max = eval_cycle_max
        self.eval_cycle_min = eval_cycle_min
        self.args = args
        self.root_path = args.root_path.replace('Battery-LLM', 'BatteryLife')
        self.seq_len = args.seq_len
        self.charge_discharge_len = args.charge_discharge_length  # The resampled length for charge and discharge curves
        self.flag = flag
        self.dataset = args.dataset if not use_target_dataset else args.target_dataset
        self.early_cycle_threshold = args.early_cycle_threshold
        self.KDE_samples = []

        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
        self.aug_helper = BatchAugmentation_battery_revised()

        self.name2domainID = json.load(open(f'/data/trf/python_works/BatteryMoE/gate_data/name2agingConditionID.json'))

        assert flag in ['train', 'test', 'val']
        if self.dataset == 'exp':
            self.train_files = split_recorder.Stanford_train_files[:3]
            self.val_files = split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            self.test_files =  split_recorder.Tongji_test_files[:2] + split_recorder.HUST_test_files[:2]
        elif self.dataset == 'Tongji':
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == 'HUST':
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == 'MATR':
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == 'SNL':
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == 'MICH':
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == 'MICH_EXP':
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == 'UL_PUR':
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == 'RWTH':
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == 'HNEI':
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == 'CALCE':
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == 'Stanford':
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == 'ISU_ILCC':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'ISU_ILCC2':
            #ISU_ILCC_delG49C1
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_delG49C1_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'XJTU':
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == 'MIX_large':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'ZN-coin':
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files 
            self.test_files = split_recorder.ZNcoin_test_files  
        elif self.dataset == 'CALB':
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files 
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == 'ZN-coin42':
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == 'ZN-coin2024':
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == 'CALB42':
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == 'CALB2024':
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == 'NAion':
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == 'NAion42':
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == 'NAion2024':
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        elif self.dataset == 'MIX_fig_cathode_NCM':
            self.train_files = split_recorder.MIX_large_cathode_NCM_train_files + split_recorder.MIX_large_cathode_NCM840610_train_files + split_recorder.MIX_large_cathode_NCM111_train_files + split_recorder.MIX_large_cathode_NCM523_train_files + split_recorder.MIX_large_cathode_NCM831107_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM_val_files + split_recorder.MIX_large_cathode_NCM840610_val_files + split_recorder.MIX_large_cathode_NCM111_val_files + split_recorder.MIX_large_cathode_NCM523_val_files + split_recorder.MIX_large_cathode_NCM831107_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM_test_files + split_recorder.MIX_large_cathode_NCM840610_test_files + split_recorder.MIX_large_cathode_NCM111_test_files + split_recorder.MIX_large_cathode_NCM523_test_files + split_recorder.MIX_large_cathode_NCM831107_test_files
        elif self.dataset == 'MIX_fig_cathode_NCA':
            self.train_files = split_recorder.MIX_large_cathode_NCA811405_train_files + split_recorder.MIX_large_cathode_NCA801505_train_files + split_recorder.MIX_large_cathode_NCA861103_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCA811405_val_files + split_recorder.MIX_large_cathode_NCA801505_val_files + split_recorder.MIX_large_cahtode_NCA861103_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCA811405_test_files + split_recorder.MIX_large_cathode_NCA801505_test_files + split_recorder.MIX_large_cathode_NCA861103_test_files
        elif self.dataset == 'MIX_fig_cathode_NCMNCA':
            self.train_files = split_recorder.MIX_large_cathode_NCM_NCA_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM_NCA_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM_NCA_test_files
        elif self.dataset == 'MIX_fig_cathode_NCMLCO':
            self.train_files = split_recorder.MIX_large_cathode_NCM422LCO_train_files
            self.val_files = split_recorder.MIX_large_cathode_NCM422LCO_val_files
            self.test_files = split_recorder.MIX_large_cathode_NCM422LCO_test_files
        elif self.dataset == 'MIX_fig_cathode_LCO':
            self.train_files = split_recorder.MIX_large_cathode_LCO_train_files
            self.val_files = split_recorder.MIX_large_cathode_LCO_val_files
            self.test_files = split_recorder.MIX_large_cathode_LCO_test_files
        elif self.dataset == 'MIX_fig_cathode_LFP':
            self.train_files = split_recorder.MIX_large_cathode_LFP_train_files
            self.val_files = split_recorder.MIX_large_carhode_LFP_val_files
            self.test_files = split_recorder.MIX_large_cathode_LFP_test_files
        elif self.dataset == 'MIX_fig_anode_graphite':
            self.train_files = split_recorder.MIX_large_anode_graphite_train_files + split_recorder.MIX_large_anode_carbon_train_files + split_recorder.MIX_large_anode_graphite_PVDF_train_files + split_recorder.MIX_large_anode_AG_train_files
            self.val_files = split_recorder.MIX_large_anode_graphite_val_files + split_recorder.MIX_large_anode_carbon_val_files + split_recorder.MIX_large_anode_graphite_PVDF_val_files + split_recorder.MIX_large_anode_AG_val_files
            self.test_files = split_recorder.MIX_large_anode_graphite_test_files + split_recorder.MIX_large_anode_carbon_test_files + split_recorder.MIX_large_anode_graphite_PVDF_test_files + split_recorder.MIX_large_anode_AG_test_files
        elif self.dataset == 'MIX_fig_anode_graphite_si':
            self.train_files = split_recorder.MIX_large_anode_graphite_si_train_files
            self.val_files = split_recorder.MIX_large_anode_graphite_si_val_files
            self.test_files = split_recorder.MIX_large_anode_graphite_si_test_files
        elif self.dataset == 'MIX_fig_format_prismatic':
            self.train_files = split_recorder.MIX_large_format_prismatic_train_files
            self.val_files = split_recorder.MIX_large_format_prismatic_val_files
            self.test_files = split_recorder.MIX_large_format_prismatic_test_files
        elif self.dataset == 'MIX_fig_format_18650':
            self.train_files = split_recorder.MIX_large_format_18650_train_files
            self.val_files = split_recorder.MIX_large_format_18650_val_files
            self.test_files = split_recorder.MIX_large_format_18650_test_files
        elif self.dataset == 'MIX_fig_format_pouch':
            self.train_files = split_recorder.MIX_large_format_pouch_train_files 
            self.val_files = split_recorder.MIX_large_format_pouch_val_files 
            self.test_files = split_recorder.MIX_large_format_pouch_test_files
        elif self.dataset == 'MIX_fig_format_502030_pouch':
            self.train_files = split_recorder.MIX_large_format_502030_pouch_train_files
            self.val_files = split_recorder.MIX_large_format_502030_pouch_val_files
            self.test_files = split_recorder.MIX_large_format_502030_pouch_test_files 
        elif self.dataset == 'MIX_fig_format_4090132_pouch':
            self.train_files = split_recorder.MIX_large_format_4090132_pouch_train_files
            self.val_files = split_recorder.MIX_large_format_4090132_pouch_val_files
            self.test_files = split_recorder.MIX_large_format_4090132_pouch_test_files
        elif self.dataset == 'MIX_fig_temp_neg5':
            self.train_files = split_recorder.MIX_large_temp_5_train_files
            self.val_files = split_recorder.MIX_large_temp_5_val_files
            self.test_files = split_recorder.MIX_large_temp_5_test_files
        elif self.dataset == 'MIX_fig_temp_15':
            self.train_files = split_recorder.MIX_large_temp_15_train_files
            self.val_files = split_recorder.MIX_large_temp_15_val_files
            self.test_files = split_recorder.MIX_large_temp_15_test_files
        elif self.dataset == 'MIX_fig_temp_20':
            self.train_files = split_recorder.MIX_large_temp_20_train_files
            self.val_files = split_recorder.MIX_large_temp_20_val_files
            self.test_files = split_recorder.MIX_large_temp_20_test_files
        elif self.dataset == 'MIX_fig_temp_23':
            self.train_files = split_recorder.MIX_large_temp_23_train_files
            self.val_files = split_recorder.MIX_large_temp_23_val_files
            self.test_files = split_recorder.MIX_large_temp_23_test_files
        elif self.dataset == 'MIX_fig_temp_25':
            self.train_files = split_recorder.MIX_large_temp_25_train_files
            self.val_files = split_recorder.MIX_large_temp_25_val_files
            self.test_files = split_recorder.MIX_large_temp_25_test_files
        elif self.dataset == 'MIX_fig_temp_30':
            self.train_files = split_recorder.MIX_large_temp_30_train_files
            self.val_files = split_recorder.MIX_large_temp_30_val_files
            self.test_files = split_recorder.MIX_large_temp_30_test_files
        elif self.dataset == 'MIX_fig_temp_35':
            self.train_files = split_recorder.MIX_large_temp_35_train_files
            self.val_files = split_recorder.MIX_large_temp_35_val_files
            self.test_files = split_recorder.MIX_large_temp_35_test_files
        elif self.dataset == 'MIX_fig_temp_45':
            self.train_files = split_recorder.MIX_large_temp_45_train_files
            self.val_files = split_recorder.MIX_large_temp_45_val_files
            self.test_files = split_recorder.MIX_large_temp_45_test_files
        elif self.dataset == 'MIX_eval':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files
            self.test_files = split_recorder.MIX_large_val_files # used for testing model performance on the validation set
        elif self.dataset == 'MIX_large_ablation_75p' and self.args.seed == 2021:
            self.train_files = split_recorder.MIX_large_reduced_train_75p_files_2021
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_75p' and self.args.seed == 42:
            self.train_files = split_recorder.MIX_large_reduced_train_75p_files_42
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_75p' and self.args.seed == 2024:
            self.train_files = split_recorder.MIX_large_reduced_train_75p_files_2024
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_50p' and self.args.seed == 2021:
            self.train_files = split_recorder.MIX_large_reduced_train_50p_files_2021
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_50p' and self.args.seed == 42:
            self.train_files = split_recorder.MIX_large_reduced_train_50p_files_42
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_50p' and self.args.seed == 2024:
            self.train_files = split_recorder.MIX_large_reduced_train_50p_files_2024
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_25p' and self.args.seed == 2021:
            self.train_files = split_recorder.MIX_large_reduced_train_25p_files_2021
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_25p' and self.args.seed == 42:
            self.train_files = split_recorder.MIX_large_reduced_train_25p_files_42
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'MIX_large_ablation_25p' and self.args.seed == 2024:
            self.train_files = split_recorder.MIX_large_reduced_train_25p_files_2024
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'Stanford_formation' and self.args.seed == 2021:
            self.train_files = split_recorder.Stanford_formation_45_train_files_2021
            self.val_files = split_recorder.Stanford_formation_45_val_files_2021
            self.test_files = split_recorder.Stanford_formation_45_test_files_2021
        elif self.dataset == 'Stanford_formation' and self.args.seed == 42:
            self.train_files = split_recorder.Stanford_formation_45_train_files_42
            self.val_files = split_recorder.Stanford_formation_45_val_files_42
            self.test_files = split_recorder.Stanford_formation_45_test_files_42
        elif self.dataset == 'Stanford_formation' and self.args.seed == 2024:
            self.train_files = split_recorder.Stanford_formation_45_train_files_2024
            self.val_files = split_recorder.Stanford_formation_45_val_files_2024
            self.test_files = split_recorder.Stanford_formation_45_test_files_2024
        else:
            raise Exception(f'{self.dataset} is not supported!')
        
        if flag == 'train':
            self.files = [i for i in self.train_files]
        elif flag == 'val':
            self.files = [i for i in self.val_files]
        elif flag == 'test':
            self.files = [i for i in self.test_files]
            self.root_path = self.root_path.replace('Battery-LLM', 'BatteryLife')
            if self.dataset == 'ZN-coin42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN42.json'))
            elif self.dataset == 'ZN-coin2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN2024.json'))
            elif self.dataset == 'CALB42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB42.json'))
            elif self.dataset == 'CALB2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB2024.json'))
            elif self.dataset == 'NAion':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2021.json'))
            elif self.dataset == 'NAion42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA42.json'))
            elif self.dataset == 'NAion2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2024.json'))
            else:
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test.json'))
            # self.unseen_seen_record = json.load(open(f'{self.root_path}/cal_for_test.json'))
        

        self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.unique_labels, self.class_labels, self.total_dataset_ids, self.total_cj_aug_charge_discharge_curves, self.total_seen_unseen_IDs, self.total_domain_ids = self.read_data()

        self.KDE_samples = copy.deepcopy(self.total_labels) if flag == 'train' else []

        self.weights = self.get_loss_weight()
        if np.any(np.isnan(self.total_charge_discharge_curves)):
            raise Exception('Nan in the data')
        if np.any(np.isnan(self.unique_labels)):
            raise Exception('Nan in the labels')
        # K-means to classify the battery life
        
        self.raw_labels = copy.deepcopy(self.total_labels)
        if flag == 'train' and label_scaler is None:
            self.label_scaler = StandardScaler()
            self.life_class_scaler = StandardScaler()
            self.label_scaler.fit(np.array(self.unique_labels).reshape(-1, 1))
            self.life_class_scaler.fit(np.array(self.class_labels).reshape(-1, 1))
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1, 1))
            self.scaled_life_classes = np.array(self.class_labels) - 1
            #self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1, 1))
        else:
            # validation set or testing set
            assert label_scaler is not None
            self.label_scaler = label_scaler
            self.life_class_scaler = life_class_scaler
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1,1))
            self.scaled_life_classes = np.array(self.class_labels) - 1
            #self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1,1))

    def get_loss_weight(self, method='KDE'):
        '''
        Get the weight for weighted loss
        method can be ['1/n', '1/log(x+1)', 'KDE']
        '''
        if self.args.weighted_loss and self.flag == 'train':
            if method == '1/n':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                weights = 1.0 / label_to_count[df["label"]].values
            elif method == '1/log(x+1)':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x)+1)
                weights = 1 / normalized_x
            elif method == 'KDE':
                # Define DenseWeight
                dw = DenseWeight(alpha=1.0)
                # Fit DenseWeight and get the weights for the 1000 samples
                dw.fit(self.KDE_samples)
                # Calculate the weight for an arbitrary target value
                weights = []
                for label in self.KDE_samples:
                    single_sample_weight = dw(label)[0]
                    weights.append(single_sample_weight)
            else:
                raise Exception('Not implemented')
            return weights
        else:
            return np.ones(len(self.total_charge_discharge_curves))

    
    def get_center_vector_index(self, file_name):
        prefix = file_name.split('_')[0]
        if prefix in ['MATR', 'HUST'] or 'LFP' in file_name:
            return 0
        else:
            return 1 
        
    def return_label_scaler(self):
        return self.label_scaler
    
    def return_life_class_scaler(self):
        return self.life_class_scaler
    
    def __len__(self):
        return len(self.total_labels)
        
    def read_data(self):
        '''
        read all data from files
        :return: x_enc, x_cycle_numbers, prompts, charge_data, discharge_data, RPT_masks, labels
        '''
    
        total_charge_discharge_curves = []
        total_curve_attn_masks = []
        total_labels = [] # RUL
        unique_labels = []
        class_labels = [] # the pseudo class for samples
        total_dataset_ids = []
        total_cj_aug_charge_discharge_curves = []
        total_seen_unseen_IDs = []
        total_domain_ids = []

        for file_name in tqdm(self.files):
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_id = datasetName2ids[file_name.split('_')[0]]
            else:
                dataset_id = datasetName2ids['MICH_EXP']

            charge_discharge_curves, attn_masks, labels, eol, cj_aug_charge_discharge_curves = self.read_samples_from_one_cell(
                file_name)
            if eol is None:
                # This battery has not reached end of life
                continue
            
            # for class_label, life_range in self.life_classes.items():
            #     if eol >= life_range[0] and eol < life_range[1]:
            #         class_label = int(class_label)
            #         class_labels += [class_label for _ in range(len(charge_discharge_curves))]
            #         break
            class_labels += [0 for _ in range(len(charge_discharge_curves))]
            

            cell_name = file_name
            domain_id = self.name2domainID[cell_name]

            total_charge_discharge_curves += charge_discharge_curves
            total_cj_aug_charge_discharge_curves += cj_aug_charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels 
            total_domain_ids += [domain_id for _ in range(len(labels))]
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)

            if self.flag == 'test' and self.dataset != 'MIX_eval':
                seen_unseen_id = self.unseen_seen_record[file_name]
                if seen_unseen_id == 'unseen':
                    total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                elif seen_unseen_id == 'seen':
                    total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                else:
                    raise Exception('Check the bug!')
            else:
                total_seen_unseen_IDs += [1 for _ in range(len(labels))] # 1 indicates seen. This is not used on training or evaluation set

        return total_charge_discharge_curves, total_curve_attn_masks, np.array(total_labels), unique_labels, class_labels, total_dataset_ids, total_cj_aug_charge_discharge_curves, total_seen_unseen_IDs, total_domain_ids

    
    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        prefix = file_name.split('_')[0]
        if prefix.startswith('MATR'):
            data =  pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data =  pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data =  pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data =  pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data =  pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            if not os.path.isdir(f'{self.root_path}/total_MICH/'):
                self.merge_MICH(f'{self.root_path}/total_MICH/')
            data =  pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data =  pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data =  pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))  
        elif prefix.startswith('UL-PUR'):
            data =  pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))  
        elif prefix.startswith('SMICH'):
            data =  pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb')) 
        elif prefix.startswith('BIT2'):
            data =  pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb')) 
        elif prefix.startswith('Tongji'):
            data =  pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            data =  pickle.load(open(f'{self.root_path}/Stanford/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data =  pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data =  pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data =  pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data =  pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data =  pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))
        
        if prefix == 'MICH':
            with open(f'{self.root_path}/Life labels/total_MICH_labels.json') as f:
                life_labels = json.load(f)
        elif prefix.startswith('Tongji'):
            file_name = file_name.replace('--', '-#')
            with open(f'{self.root_path}/Life labels/Tongji_labels.json') as f:
                life_labels = json.load(f)
        else:
            with open(f'{self.root_path}/Life labels/{prefix}_labels.json') as f:
                life_labels = json.load(f)
        if file_name in life_labels:
            eol = life_labels[file_name]
        else:
            eol = None
        return data, eol
    
    def read_cell_df(self, file_name):
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, basic_prompt, eol
        '''
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]
        
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
            
        cycle_data = data['cycle_data'] # list of cycle data dict
            
        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah']<0] = np.nan # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah']<0] = np.nan
            cycle_df.bfill(inplace=True) # deal with NaN
            total_cycle_dfs.append(cycle_df)
            
            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break
            
        df = pd.concat(total_cycle_dfs)
        # obtain the charge and discahrge curves
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold, nominal_capacity)
        cj_aug_charge_discharge_curves, fm_aug_charge_discharge_curves  = self.aug_helper.batch_aug(charge_discharge_curves)

        return df, charge_discharge_curves, eol, nominal_capacity, cj_aug_charge_discharge_curves
    
        
    def read_samples_from_one_cell(self, file_name):
        '''
        read all samples using this function
        :param file_name: which file needs to be read
        :return: history_sohs, future_sohs, masks, cycles, prompts, charge_data, discharge_data and RPT_masks in each sample
        '''

        df, charge_discharge_curves_data, eol, nominal_capacity, cj_aug_charge_discharge_curves = self.read_cell_df(file_name)
        if df is None or eol<=self.early_cycle_threshold:
            return None, None, None, None, None

        # the charge and discharge data
        charge_discharge_curves = []  # [N, seq_len, fix_charge_resample_len]
        total_cj_aug_charge_discharge_curves = []
        attn_masks = []
        labels = []
        # get the early-life data
        early_charge_discharge_curves_data = charge_discharge_curves_data[:self.early_cycle_threshold]
        early_cj_aug_charge_discharge_curves = cj_aug_charge_discharge_curves[:self.early_cycle_threshold]
        if np.any(np.isnan(early_charge_discharge_curves_data)):
            raise Exception(f'Failure in {file_name} | Early data contains NaN! Cycle life is {eol}!')
        for i in range(self.seq_len, self.early_cycle_threshold+1):
            if i >= eol:
                # If we encounter a battery whose cycle life is even smaller than early_cycle_threhold
                # We should not include the eol cycle data
                break
            
            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1 # set 1 not to mask
            
            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if i <= self.eval_cycle_max and i >= self.eval_cycle_min:
                    # Only keep the val and test samples that satisfy the eval_cycle
                    pass
                else:
                    continue
            

            # tmp_prompt = basic_prompt
            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            total_cj_aug_charge_discharge_curves.append(early_cj_aug_charge_discharge_curves)
            attn_masks.append(tmp_attn_mask)

        return charge_discharge_curves, attn_masks, labels, eol, total_cj_aug_charge_discharge_curves

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []
        unique_cycles = df['cycle_number'].unique()
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle in range(1, early_cycle_threshold+1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]
                
                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records/nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values
                time_in_s_records = cycle_df['time_in_s'].values

                cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C<=-0.01) 
                discharge_end_index = cutoff_voltage_indices[0][-1]
                
                # tmp_discharge_capacity_records = max(charge_capacity_records) - discharge_capacity_records
                if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_35', 'CALB_45']:
                    # Every cycle first discharge and then charge
                    #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]
                    
                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity
                    
                    charge_voltages = charge_voltages[np.abs(charge_current_in_C)>0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C)>0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C)>0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C)>0.01]
                else:
                    # Every cycle first charge and then discharge
                    #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity
                    
                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C)>0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C)>0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C)>0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C)>0.01]
                    
                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]
                
                # try:
                #     discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curvesv2(discharge_voltages, discharge_currents, discharge_capacities)
                #     charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curvesv2(charge_voltages, charge_currents, charge_capacities)
                # except:
                #     print('file_name', file_name, cycle)

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(charge_voltages, charge_currents, charge_capacities)


                
                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)
                
                voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(voltage_records) # normalize using the cutoff voltage
                current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the capacity
                
                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))
              
        curves = np.concatenate(curves, axis=0) # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or dicharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interploted records
        '''
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages)+1)
        interp_bases = np.linspace(1, len(voltages)+1, num=charge_discharge_len,
                                        endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery
    

    def __getitem__(self, index):
        sample = {
                'cycle_curve_data': torch.Tensor(self.total_charge_discharge_curves[index]),
                'curve_attn_mask': torch.Tensor(self.total_curve_attn_masks[index]),
                'labels': self.total_labels[index],
                'life_class': self.class_labels[index],
                'scaled_life_class': self.scaled_life_classes[index],
                'weight': self.weights[index],
                'dataset_id': self.total_dataset_ids[index],
                'cj_cycle_curve_data': self.total_cj_aug_charge_discharge_curves[index],
                'seen_unseen_id': self.total_seen_unseen_IDs[index],
                'domain_ids': self.total_domain_ids[index]
            }
        return sample
    
    def read_train_labels(self, train_files):
        train_labels = []
        for file_name in train_files:
            prefix = file_name.split('_')[0]
            if prefix == 'MICH':
                with open(f'{self.root_path}/total_MICH_labels.json') as f:
                    life_labels = json.load(f)
            elif prefix.startswith('Tongji'):
                with open(f'{self.root_path}/Tongji_labels.json') as f:
                    life_labels = json.load(f)
            else:
                with open(f'{self.root_path}/{prefix}_labels.json') as f:
                    life_labels = json.load(f)
            if file_name in life_labels:
                eol = life_labels[file_name]
            else:
                continue
            train_labels.append(eol)
        return train_labels

    def get_RPT_str(self, RPT_masks, cycle_numbers):
        RPT_masks = np.array(RPT_masks)
        cycle_numbers = np.array(cycle_numbers)
        
        if np.all(RPT_masks==1):
            prompt = 'Described operating condition is used in all cycles.'
        else:
            tmp_RPT_cycles = []
            tmp_normal_cycles = []
            sample_RPT_masks = RPT_masks
            for index, RPT_mask in enumerate(sample_RPT_masks):
                if RPT_mask == 0:
                    tmp_RPT_cycles.append(cycle_numbers[index])
                elif RPT_mask == 1:
                    tmp_normal_cycles.append(cycle_numbers[index])
            prompt = f'Describned operating condition is used in {tmp_normal_cycles} cycles, wheras cycles {tmp_RPT_cycles} are conducted using other operating conditions. '       
        return prompt
    
    def merge_MICH(self, merge_path):
        os.makedirs(merge_path)
        source_path1 = f'{self.root_path}/MICH/'
        source_path2 = f'{self.root_path}/MICH_EXP/'
        source1_files = [i for i in os.listdir(source_path1) if i.endswith('.pkl')]
        source2_files = [i for i in os.listdir(source_path2) if i.endswith('.pkl')]
        target_path = f'{self.root_path}/total_MICH/'

        for file in source1_files:
            shutil.copy(source_path1 + file, target_path)
        for file in source2_files:
            shutil.copy(source_path2 + file, target_path)
