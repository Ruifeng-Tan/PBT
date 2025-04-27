import os
import random
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
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

# Temperature assignment
# Only Li-ion
# temperature2mask = {
#     -5.0: [0],
#     15.0: [1],
#     20.0: [2],
#     23.0: [3],
#     25.0: [4,5],
#     30.0: [6,7,8,9,10,11],
#     35.0: [12],
#     45.0: [13],
#     55.0: [14]
# }

# Li-ion, Na-ion, Zn-ion, CALB
# temperature2mask = {
#     -5.0: [0],
#     0.0: [1],
#     15.0: [2],
#     20.0: [3],
#     23.0: [4],
#     25.0: [5,6,7,8],
#     30.0: [9,10,11,12,13,14],
#     35.0: [15],
#     45.0: [16],
#     55.0: [17]
# }

# We assign each temperature to the three neighboring temperatures
# Only Li-ion
# temperature2mask = {
#     -5.0: [0,1,2],
#     15.0: [0,1,2],
#     20.0: [1,2,3],
#     23.0: [2,3,4,5],
#     25.0: [3,4,5,6,7,8,9,10,11],
#     30.0: [4,5,6,7,8,9,10,11,12],
#     35.0: [6,7,8,9,10,11,12,13],
#     45.0: [12,13,14],
#     55.0: [12,13,14]
# }
# Li-ion, Na-ion, Zn-ion, CALB
# temperature2mask = {
#     -5.0: [0,1,2],
#     0.0: [0,1,2],
#     15.0: [1,2,3],
#     20.0: [2,3,4],
#     23.0: [3,4,5,6,7,8],
#     25.0: [4,5,6,7,8,9,10,11,12,13,14],
#     30.0: [5,6,7,8,9,10,11,12,13,14,15],
#     35.0: [9,10,11,12,13,14,15,16],
#     45.0: [15,16,17],
#     55.0: [15,16,17]
# }

# assign experts according to formats
# only Li-ion
# format2mask = {
#     'prismatic': [0],
#     'cylindrical': [1,2,3,4,5,6],
#     'polymer': [7,8,9],
#     'pouch': [10]
# } 

# Li-ion, Na-ion, Zn-ion, CALB
# format2mask = {
#     'prismatic': [0],
#     'cylindrical': [1,2,3,4,5,6],
#     'polymer': [7,8,9],
#     'pouch': [10],
#     'coin': [11]
# } 

# only Li-ion
# cathodes2mask = {
#     'LFP': [0,1,2],
#     'NCA': [3],
#     'NCM': [4,5,6,7,8,9],
#     'LCO': [10],
#     'NCA_NCM': [3,4,5,6,7,8,9],
#     'NCM_NCA': [3,4,5,6,7,8,9],
#     'LCO_NCM': [4,5,6,7,8,9,10],
#     'NCM_LCO': [4,5,6,7,8,9,10]
# }

# Li-ion, Na-ion, Zn-ion, CALB
# cathodes2mask = {
#     'LFP': [0,1,2],
#     'NCA': [3],
#     'NCM': [4,5,6,7,8,9],
#     'LCO': [10],
#     'NCA_NCM': [3,4,5,6,7,8,9],
#     'NCM_NCA': [3,4,5,6,7,8,9],
#     'LCO_NCM': [4,5,6,7,8,9,10],
#     'NCM_LCO': [4,5,6,7,8,9,10],
#     'MnO2': [11],
#     'Unknown': [12]
# }

# Anode
# Only Li-ion
# anode2mask = {
#     'graphite': [0,1,2,3,4,5,6,7,8,9],
#     'graphite/Si': [10]
# }

# Li-ion, Na-ion, Zn-ion, CALB
# anode2mask = {
#     'graphite': [0,1,2,3,4,5,6,7,8,9],
#     'graphite/Si': [10],
#     'zinc metal': [11],
#     'Unknown': [12]
# }

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
    combined_masks = torch.vstack([i['combined_mask'] for i in samples])

    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])

    labels = torch.Tensor([i['labels'] for i in samples])

    weights = torch.Tensor([i['weight'] for i in samples])
    
    DKP_embeddings = torch.vstack([i['DKP_embedding'] for i in samples])
    dataset_ids = torch.Tensor([i['dataset_id'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])
    return cycle_curve_data, curve_attn_mask, labels, weights, dataset_ids, seen_unseen_ids, DKP_embeddings, cathode_masks, temperature_masks, format_masks, anode_masks, combined_masks

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
    combined_masks = torch.vstack([i['combined_mask'] for i in samples])


    DKP_embeddings = torch.vstack([i['DKP_embedding'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])

    return cycle_curve_data, curve_attn_mask, labels, weights, file_names, DKP_embeddings, seen_unseen_ids, cathode_masks, temperature_masks, format_masks, anode_masks, combined_masks

# BatterLifeLLM dataloader
class Dataset_BatteryLifeLLM_original(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, tokenizer=None, eval_cycle_max=None, eval_cycle_min=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, temperature2mask=None, format2mask=None, cathodes2mask=None, anode2mask=None, use_target_dataset=False):
        '''
        init the Dataset_BatteryFormer class
        :param args:model parameters
        :param flag:including train, val, test
        :param scaler:scaler or not
        '''
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

        self.temperature2mask = temperature2mask
        self.format2mask = format2mask
        self.cathodes2mask = cathodes2mask
        self.anode2mask = anode2mask


        self.label_prompts_vectors = {}
        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
        self.aug_helper = BatchAugmentation_battery_revised()
        assert flag in ['train', 'test', 'val']
        if self.dataset == 'exp':
            self.train_files = split_recorder.Stanford_train_files[:3] + split_recorder.Tongji_train_files[:2]
            self.val_files = split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            self.test_files =  split_recorder.Tongji_test_files[:2] + split_recorder.HUST_test_files[:1]
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
        elif self.dataset == 'MIX_CALB':
            self.train_files = split_recorder.MIX_CALB_train_files
            self.val_files = split_recorder.MIX_CALB_val_files
            self.test_files = split_recorder.MIX_CALB_test_files
        elif self.dataset == 'MIX_CALB42':
            self.train_files = split_recorder.MIX_CALB42_train_files
            self.val_files = split_recorder.MIX_CALB42_val_files
            self.test_files = split_recorder.MIX_CALB42_test_files
        elif self.dataset == 'MIX_CALB2024':
            self.train_files = split_recorder.MIX_CALB2024_train_files
            self.val_files = split_recorder.MIX_CALB2024_val_files
            self.test_files = split_recorder.MIX_CALB2024_test_files
        elif self.dataset == 'MIX_ZN':
            self.train_files = split_recorder.MIX_ZN_train_files
            self.val_files = split_recorder.MIX_ZN_val_files
            self.test_files = split_recorder.MIX_ZN_test_files
        elif self.dataset == 'MIX_ZN42':
            self.train_files = split_recorder.MIX_ZN42_train_files
            self.val_files = split_recorder.MIX_ZN42_val_files
            self.test_files = split_recorder.MIX_ZN42_test_files
        elif self.dataset == 'MIX_ZN2024':
            self.train_files = split_recorder.MIX_ZN2024_train_files
            self.val_files = split_recorder.MIX_ZN2024_val_files
            self.test_files = split_recorder.MIX_ZN2024_test_files
        elif self.dataset == 'MIX_NA':
            self.train_files = split_recorder.MIX_NA_train_files
            self.val_files = split_recorder.MIX_NA_val_files
            self.test_files = split_recorder.MIX_NA_test_files
        elif self.dataset == 'MIX_NA42':
            self.train_files = split_recorder.MIX_NA42_train_files
            self.val_files = split_recorder.MIX_NA42_val_files
            self.test_files = split_recorder.MIX_NA42_test_files
        elif self.dataset == 'MIX_NA2024':
            self.train_files = split_recorder.MIX_NA2024_train_files
            self.val_files = split_recorder.MIX_NA2024_val_files
            self.test_files = split_recorder.MIX_NA2024_test_files 
        elif self.dataset == 'MIX_all':
            self.train_files = split_recorder.MIX_all_2021_train_files
            self.val_files = split_recorder.MIX_all_2021_val_files
            self.test_files = split_recorder.MIX_all_2021_test_files 
        elif self.dataset == 'MIX_all2024':
            self.train_files = split_recorder.MIX_all_2024_train_files
            self.val_files = split_recorder.MIX_all_2024_val_files
            self.test_files = split_recorder.MIX_all_2024_test_files 
        elif self.dataset == 'MIX_all42':
            self.train_files = split_recorder.MIX_all_42_train_files
            self.val_files = split_recorder.MIX_all_42_val_files
            self.test_files = split_recorder.MIX_all_42_test_files 
        
        # load the prompt embedding
        if self.seed == 2021:
            if not self.args.use_PCA:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all.pkl', 'rb'))
            else:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all_pca.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all_pca.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all_pca.pkl', 'rb'))
        elif self.seed == 2024:
            if not self.args.use_PCA:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all2024.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all2024.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all2024.pkl', 'rb'))
            else:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all2024_pca.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all2024_pca.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all2024_pca.pkl', 'rb'))
        elif self.seed == 42:
            if not self.args.use_PCA:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all42.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all42.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all42.pkl', 'rb'))
            else:
                train_part = pickle.load(open(f'{self.root_path}/training_DKP_embed_all42_pca.pkl', 'rb'))
                val_part = pickle.load(open(f'{self.root_path}/validation_DKP_embed_all42_pca.pkl', 'rb'))
                test_part = pickle.load(open(f'{self.root_path}/testing_DKP_embed_all42_pca.pkl', 'rb'))
        else:
            raise Exception('Plases generate the prompt emebddigns for this seed.')

        self.cellName_prompt = train_part | val_part | test_part
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

        self.total_prompts, self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.unique_labels, self.total_dataset_ids, self.total_center_vector_indices, self.total_file_names, self.total_cluster_labels, self.total_DKP_embeddings, self.total_seen_unseen_IDs, self.total_cathode_expert_masks, self.total_temperature_experts_masks, self.total_format_expert_masks, self.total_anode_expert_masks, self.total_combined_expert_masks = self.read_data()
        
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
    
        total_prompts = []
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
        total_combined_expert_masks = []

        total_DKP_embeddings = []
        total_cluster_labels = []

        for file_name in tqdm(self.files):
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_id = datasetName2ids[file_name.split('_')[0]]
            else:
                dataset_id = datasetName2ids['MICH_EXP']


            # center_vector_index = self.get_center_vector_index(file_name)

            prompts, charge_discharge_curves, attn_masks, labels, eol = self.read_samples_from_one_cell(
                file_name)

            if prompts is None:
                # This battery has not reached end of life
                continue

            if file_name in self.cathode_json:
                cathodes = self.cathode_json[file_name]
                cathodes = '_'.join(cathodes)
                cathode_index = self.cathodes2mask[cathodes] 
                cathode_mask = np.zeros(self.cathode_experts) # 1 indicates activated
                cathode_mask[cathode_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the cathodes.json. We suggest the user to set the cathode in the cathodes.json and manually assign the expert'
                'using the cathodes2mask based on domain knowledge. When it is not possible to know the cathode or to manually assign the cathode, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                cathode_mask = np.ones(self.cathode_experts) # assign according to the learned parameters
                # cathode_mask = np.zeros(self.cathode_experts) # only use the general experts

            cathode_mask = list(cathode_mask)*self.args.scale_factor

            if file_name in self.temperature_json:
                temperatures = self.temperature_json[file_name]
                temperature_index = self.temperature2mask[temperatures]
                temperature_mask = np.zeros(self.temperature_experts)
                temperature_mask[temperature_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the temperatures.json. We suggest the user to set the temperature in the temperatures.json and manually assign the expert'
                'using the temperature2mask based on domain knowledge. When it is not possible to know the temperature or to manually assign the temperature, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                temperature_mask = np.ones(self.temperature_experts) # assign according to the learned parameters
                # temperature_mask = np.zeros(self.temperature_experts) # only use the general experts

            temperature_mask = list(temperature_mask)*self.args.scale_factor

            if file_name in self.format_json:
                format = self.format_json[file_name][0]
                format_index = self.format2mask[format]
                format_mask = np.zeros(self.format_experts)
                format_mask[format_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the formats.json. We suggest the user to set the format in the formats.json and manually assign the expert'
                'using the format2mask based on domain knowledge. When it is not possible to know the format or to manually assign the format, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                format_mask = np.ones(self.format_experts) # assign according to the learned parameters
                # format_mask = np.zeros(self.format_experts) # only use the general experts
            format_mask = list(format_mask)*self.args.scale_factor

            if file_name in self.anode_json:
                anode = self.anode_json[file_name][0]
                if anode == 'graphite' or anode == 'artificial graphite' or anode == 'carbon':
                    anode = 'graphite' # we assume other anodes are graphite
                anode_index = self.anode2mask[anode]
                anode_mask = np.zeros(self.anode_experts)
                anode_mask[anode_index] = 1
            else:
                raise Exception(f'The {file_name} is not shown in the formats.json. We suggest the user to set the format in the formats.json and manually assign the expert'
                'using the format2mask based on domain knowledge. When it is not possible to know the format or to manually assign the format, you can consider commenting this Exception and then BatteryMoE will assign'
                'the expert for you.')
                anode_mask = np.ones(self.anode_experts) # assign according to the learned parameters
                # anode_mask = np.zeros(self.anode_experts) # only use the general experts
            anode_mask = list(anode_mask)*self.args.scale_factor

            combined_expert_mask = cathode_mask + temperature_mask + format_mask + anode_mask

            cell_name = file_name.split('.pkl')[0]
            if self.flag == 'train':
                cluster_label = -1 # not used. Should be removed
            else:
                cluster_label = -1 # The cluster labels of validation or testing samples are unknown
            DKP_embedding = self.cellName_prompt[cell_name]

            
            total_prompts += prompts

            total_charge_discharge_curves += charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels 
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            total_file_names += [file_name for _ in range(len(labels))]
            total_cluster_labels += [cluster_label for _ in range(len(labels))]
            total_DKP_embeddings += [DKP_embedding for _ in range(len(labels))]
            total_cathode_expert_masks += [cathode_mask for _ in range(len(labels))]
            total_format_expert_masks += [format_mask for _ in range(len(labels))]
            total_temperature_experts_masks += [temperature_mask for _ in range(len(labels))]
            total_anode_expert_masks += [anode_mask for _ in range(len(labels))]
            total_combined_expert_masks += [combined_expert_mask for _ in range(len(labels))]
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)
            unique_labels.append(eol)
            if self.flag == 'test':
                seen_unseen_id = self.unseen_seen_record[file_name]
                if seen_unseen_id == 'unseen':
                    total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                elif seen_unseen_id == 'seen':
                    total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                else:
                    raise Exception('Check the bug!')
            else:
                total_seen_unseen_IDs += [1 for _ in range(len(labels))] # 1 indicates seen. This is not used on training or evaluation set

        return total_prompts, total_charge_discharge_curves, total_curve_attn_masks, np.array(total_labels), unique_labels, total_dataset_ids, total_center_vector_indices, total_file_names, total_cluster_labels, total_DKP_embeddings, total_seen_unseen_IDs, total_cathode_expert_masks, total_temperature_experts_masks, total_format_expert_masks, total_anode_expert_masks, total_combined_expert_masks

    
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
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold, nominal_capacity)
        # cj_aug_charge_discharge_curves, fm_aug_charge_discharge_curves  = self.aug_helper.batch_aug(charge_discharge_curves)

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
            return None, None, None, None, None

        # the charge and discharge data
        prompts = []
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
            
            if 'Instruct' in self.args.LLM_path:
                # Llama-instruct
                messages = [
                    {"role": "system", "content": "You are an expert in predicting battery cycle life."},
                    {"role": "user", "content": tmp_prompt}
                ]

                tmp_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                tmp_prompt = '<|begin_of_text|>' + tmp_prompt

            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            prompts.append(tmp_prompt)
            attn_masks.append(tmp_attn_mask)

        return prompts, charge_discharge_curves, attn_masks, labels, eol

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
                'combined_mask': torch.Tensor(self.total_combined_expert_masks[index]),
                'DKP_embedding': torch.from_numpy(self.total_DKP_embeddings[index]),
                'cluster_label': self.total_cluster_labels[index],
                'file_name': self.total_file_names[index],
                'seen_unseen_id': self.total_seen_unseen_IDs[index]
            }
        return sample
 