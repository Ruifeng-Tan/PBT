'''
最终版本的BatteryMoE
'''
import torch
import copy
import math
import pickle
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
import transformers
from scipy import signal
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel, AutoConfig, Phi3Config
from transformers import PreTrainedModel, BitsAndBytesConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryLifeConfig
from BatteryLifeLLMUtils.output_BatteryLifeLLM import BatteryLifeCausalLMOutputWithPast
from layers.Embed import PositionalEmbedding
from layers.Transformer_EncDec import PBTEncoder, PBTEncoderLayer, ConvLayer, RMSEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
from layers.Embed import TokenEmbedding, DataEmbedding
from layers.distributional_router_encoder import DistributionRouter, PatternRouterMLP
from layers.MOE_dispatcher import MOEDispatcher
from utils.tools import sample_top_p
from utils.augmentation import Cutout_jitter_aug, BatchAugmentation_battery
import numpy as np
from typing import List, Literal, Optional, Tuple, TypedDict
import torch.nn.functional as F
from transformers import AwqConfig, AutoModelForCausalLM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from layers.MLPs import PBTMLPBlock, PBTCyclePatch
transformers.logging.set_verbosity_error() 

    
class OutputHead(nn.Module):
    def __init__(self, ec_config):
        super(OutputHead, self).__init__()
        ec_config = ec_config.get_configs()
        self.d_llm = ec_config.d_llm
        self.d_ff = ec_config.d_ff
        self.d_model = ec_config.d_model
        self.early_cycle_threshold = ec_config.early_cycle_threshold
        self.drop_rate = ec_config.dropout
        self.n_heads = ec_config.n_heads
        self.projection = nn.Sequential(nn.Linear(self.d_model, ec_config.output_num))
        
    def forward(self, llm_out):
        '''
        llm_out: [N, L, d_llm]
        llm_attn_mask: [N, L]
        curve_attn_mask: [N, L]
        '''
        out = self.projection(llm_out)
        
        return out, llm_out, llm_out

class Model(nn.Module):
    '''
    The load balancing loss is from the paper "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity".
    '''
    def __init__(self, battery_life_config):
        super(Model, self).__init__()
        configs = battery_life_config.ec_config.get_configs()
        self.configs = configs
        self.task_name = configs.task_name
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.n_heads = configs.n_heads
        self.charge_discharge_length = configs.charge_discharge_length

        # loading llama-2-7b model configs
        # 加载llama-2-7B模型参数
        self.tokenizer = AutoTokenizer.from_pretrained(
            configs.LLM_path,
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True, 
            pad_token='<|endoftext|>'
        )
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.d_model = configs.d_model
        self.d_llm = configs.d_llm
        self.tokenizer.padding_side = 'right' # set the padding side
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.moe_layers = configs.e_layers+configs.d_layers
        self.drop_rate = configs.dropout
        self.activation = configs.activation
        self.cathode_experts = configs.cathode_experts
        self.temperature_experts = configs.temperature_experts
        self.format_experts = configs.format_experts
        self.anode_experts = configs.anode_experts
        self.num_general_experts = configs.num_general_experts
        self.ion_experts = configs.ion_experts
        self.num_views = configs.num_views
        self.down_sample_ratio = configs.down_sample_ratio

        self.cathode_split = self.cathode_experts
        self.num_experts = self.cathode_experts + self.temperature_experts + self.format_experts + self.anode_experts

        self.gate_d_ff = configs.gate_d_ff
        self.dk_factor = configs.dk_factor
        self.gate_domain_knowledge_neurons = self.num_experts * configs.dk_factor

        self.pca_scaler = pickle.load(open(configs.pca_path, 'rb'))

        assert self.d_ff >= self.gate_domain_knowledge_neurons, Exception('The gate neurons should be no less than the domain-knowledge neurons')
        self.gate = nn.Sequential(nn.Linear(self.d_llm, self.gate_d_ff, bias=False))
        gate_dim = self.gate_d_ff
        self.split_dim = self.d_model // self.num_views
        self.d_ff_scale_factor = configs.d_ff_scale_factor
        
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = PBTCyclePatch(self.charge_discharge_length*3, self.d_ff, gate_dim, self.d_model, self.drop_rate, self.activation)
        self.intra_MLP = nn.ModuleList([PBTMLPBlock(self.d_model, self.d_ff, gate_dim, self.drop_rate, self.activation) for _ in range(configs.e_layers)])
        self.pe = PositionalEmbedding(self.d_model)
        self.inter_TransformerEncoder = PBTEncoder(
            [
                PBTEncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    gate_dim,
                    configs.d_ff,
                    dropout=configs.dropout
                ) for l in range(configs.d_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.d_model) 
        self.regression_head = OutputHead(battery_life_config.ec_config)


    def forward(self, cycle_curve_data, curve_attn_mask, 
                attention_mask: Optional[torch.Tensor] = None,
                DKP_embeddings: Optional[torch.FloatTensor] = None,
                cathode_masks: Optional[torch.Tensor] = None,
                temperature_masks: Optional[torch.Tensor] = None,
                format_masks: Optional[torch.Tensor] = None,
                anode_masks: Optional[torch.Tensor] = None,
                ion_type_masks: Optional[torch.Tensor] = None,
                combined_masks: Optional[torch.Tensor] = None,
                SOH_trajectory: Optional[torch.Tensor] = None,
                CE_trajectory: Optional[torch.Tensor] = None,
                use_aug: bool = False,
                return_embedding: bool=False,
                use_view_experts: bool=True
                ):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]. 0 indicates masked
        '''
        # process the charge&discharge data
        B, L, num_var, fixed_len = cycle_curve_data.shape[0], cycle_curve_data.shape[1], cycle_curve_data.shape[2], cycle_curve_data.shape[3]
        cycle_curve_data, curve_attn_mask = cycle_curve_data.to(torch.bfloat16), curve_attn_mask.to(torch.bfloat16)
        DKP_embeddings = DKP_embeddings.to(torch.bfloat16)
        if use_aug:
            random_point_num =  int(fixed_len*self.down_sample_ratio)

            flatten_cycle_curve_data = cycle_curve_data.reshape(B, L*num_var, -1)
            flatten_cycle_curve_data = flatten_cycle_curve_data.transpose(1, 2) # [B, fixed_len, L*num_var]

            flatten_cycle_curve_data = flatten_cycle_curve_data.expand(3, -1, -1, -1)  # [3, B, fixed_len, L*num_var]
            flatten_cycle_curve_data = flatten_cycle_curve_data.reshape(3 * B, fixed_len, flatten_cycle_curve_data.shape[-1])  # [3*B, fixed_len, L*num_var]
            flatten_cycle_curve_data[B:] = self.CD_Crop_augmentation(flatten_cycle_curve_data[B:], random_point_num) # add noise
            flatten_cycle_curve_data = flatten_cycle_curve_data.transpose(1, 2) # [2*B, L*num_var, fixed_len]
            cycle_curve_data = flatten_cycle_curve_data.reshape(3*B, L, num_var, fixed_len)

            # cycle_curve_data = torch.cat([cycle_curve_data, aug_cycle_curve_data], dim=0)
            curve_attn_mask = curve_attn_mask.unsqueeze(0).expand(3, -1, -1).reshape(3*B, -1)
            DKP_embeddings = DKP_embeddings.unsqueeze(0).expand(3, -1, -1).reshape(3*B, -1)
            combined_masks = combined_masks.unsqueeze(0).expand(3, -1, -1).reshape(3*B, -1)
            ion_type_masks = ion_type_masks.unsqueeze(0).expand(3, -1, -1).reshape(3*B, -1)


        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros



        DKP_embeddings = self.gate(DKP_embeddings) # [B, gate_d_ff]
        DKP_embeddings = DKP_embeddings.unsqueeze(1)
        knowledge_mask = combined_masks.repeat_interleave(dim=1, repeats=self.dk_factor)
        if self.d_ff > self.gate_domain_knowledge_neurons:
            knowledge_mask = torch.nn.functional.pad(knowledge_mask, (0, self.d_ff - self.gate_domain_knowledge_neurons), value=1.0)
            knowledge_mask = knowledge_mask.unsqueeze(1) # [B, 1, d_ff]

        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, early_cycle, fixed_len * num_var]
        out = self.intra_embed(cycle_curve_data, DKP_embeddings, knowledge_mask)
        for i in range(self.e_layers):
            out = self.intra_MLP[i](out, DKP_embeddings, knowledge_mask) # [B, early_cycle, d_model]

        out = self.pe(out) + out
        attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        attn_mask = torch.repeat_interleave(attn_mask, attn_mask.shape[-1], dim=1) # [B, L, L]
        attn_mask = attn_mask.unsqueeze(1) # [B, 1, L, L]
        attn_mask = attn_mask==0 # set True to mask
        out, attns = self.inter_TransformerEncoder(out, attn_mask=attn_mask, DKP_embeddings=DKP_embeddings, knowledge_mask=knowledge_mask)

        lengths = torch.sum(curve_attn_mask, dim=1).cpu() # [N]
        idx = (torch.as_tensor(lengths, device=out.device, dtype=torch.long) - 1).view(-1, 1).expand(
            len(lengths), out.size(2))
        idx = idx.unsqueeze(1)
        out = out.gather(1, idx).squeeze(1) # [B, D]

        out = self.norm(out)
        preds, embeddings, feature_llm_out = self.regression_head(out)

        preds = preds.float()
        embeddings = embeddings.float()
        return preds[:B], None, embeddings[B:], feature_llm_out, None, None, 0 , 0

    def create_causal_mask(self, B, seq_len):
        '''
        return:
            casual mask: [B, L, L]. 0 indicates masked.
        '''
        # Create a lower triangular matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask

    def CD_Crop_augmentation(self, X: torch.Tensor, random_point_num: int) -> torch.Tensor:
        """
        Resamples the input tensor X by selecting random points (including start and end) and interpolating back to original length.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, L, D].
            random_point_num (int): Number of points to randomly select, must be at least 2.
        
        Returns:
            torch.Tensor: Resampled tensor of shape [B, L, D].
        """
        B, L, D = X.shape
        charge_X = X[:, :L//2]
        discharge_X = X[:, L//2:]
        aug_charge_X = self.Crop(charge_X, random_point_num=random_point_num//2)
        aug_discharge_X = self.Crop(discharge_X, random_point_num=random_point_num//2)

        X = torch.cat([aug_charge_X, aug_discharge_X], dim=1) # [B, random_point_num, D]
        return X

    def Crop(self, X: torch.Tensor, random_point_num: int) -> torch.Tensor:
        """
        Resamples the input tensor X by selecting random points (including start and end) and interpolating back to original length.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, L, D].
            random_point_num (int): Number of points to randomly select, must be at least 2.
        
        Returns:
            torch.Tensor: Resampled tensor of shape [B, L, D].
        """
        B, L, D = X.shape
        K = random_point_num

        if K < 2:
            raise ValueError("random_point_num must be at least 2")
        if L < 2:
            raise ValueError("L must be at least 2")

        # Step 1: Generate indices including 0 and L-1, plus random middle points
        middle = L - 2  # Possible middle indices count (1 to L-2)
        k_middle = K - 2

        if k_middle > 0:
            if middle < k_middle:
                raise ValueError(f"Cannot select {k_middle} middle points when middle={middle}")

            # Generate random middle indices without replacement
            rand_vals = torch.rand(B, middle, device=X.device)
            middle_indices = rand_vals.argsort(dim=1)[:, :k_middle]  # [B, k_middle]
            middle_indices += 1  # Shift to range 1 to L-2 (inclusive)
        else:
            raise Exception(f'random_point_num is {random_point_num}. You should set it larger than 2!')

        # Combine with start and end indices
        zeros = torch.zeros(B, 1, dtype=torch.long, device=X.device) # start indices
        ends = (L - 1) * torch.ones(B, 1, dtype=torch.long, device=X.device) # end indices
        indices = torch.cat([zeros, middle_indices, ends], dim=1)
        indices, _ = torch.sort(indices, dim=1)  # Sort indices for each batch

        # Step 2: Gather the selected points
        selected_X = torch.gather(X, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        selected_X = selected_X.transpose(1, 2) # [B, D, random_point_num]

        # Step 3: do the linear interpolation
        interpolated = F.interpolate(
                selected_X,
                size=L,
                mode='linear',
                align_corners=True
            )
        
        return interpolated.transpose(1, 2)
