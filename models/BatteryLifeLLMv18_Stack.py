'''
router的score使用多个Linear生成多个Input的score，然后score相加
'''
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
import transformers
from math import sqrt
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel, AutoConfig, Phi3Config
from transformers import PreTrainedModel, BitsAndBytesConfig
from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryLifeConfig
from BatteryLifeLLMUtils.output_BatteryLifeLLM import BatteryLifeCausalLMOutputWithPast
from layers.Embed import PatchEmbeddingTimeLLM, PositionalEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, RMSEncoderLayer, MyEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
from layers.Embed import TokenEmbedding, DataEmbedding
from layers.fusion import GatedFusion
from layers.distributional_router_encoder import PatternRouter_two, PatternRouter_three, PatternRouter_Imp_three
from layers.MOE_dispatcher import MOEDispatcher
from utils.tools import sample_top_p
from utils.augmentation import Cutout_jitter_aug, BatchAugmentation_battery
import numpy as np
from typing import List, Literal, Optional, Tuple, TypedDict
import torch.nn.functional as F
from transformers import AwqConfig, AutoModelForCausalLM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
transformers.logging.set_verbosity_error()
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.act = nn.ReLU() if activation=='relu' else nn.GELU()
        self.out_linear = nn.Linear(hidden_dim, in_dim)
        self.ln = nn.LayerNorm(in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out
    
class BatteryLifeLLM(PreTrainedModel):
    config_class = BatteryLifeConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        # important: this ported version of LlavaNext isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava_next should serve for that purpose
        # std = (
        #     self.config.initializer_range
        #     if hasattr(self.config, "initializer_range")
        #     else self.config.text_config.initializer_range
        # )

        # if hasattr(module, "class_embedding"):
        #     module.class_embedding.data.normal_(mean=0.0, std=std)

        # if isinstance(module, (nn.Linear, nn.Conv2d)):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        pass

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

class FlattenIntraCycleMoELayer(nn.Module):
    def __init__(self, configs):
        super(FlattenIntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.num_experts = configs.num_experts
        self.top_k = configs.topK
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_experts)])
        self.general_expert = nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model))
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouter_two(self.d_llm, 1, self.d_ff, self.num_experts)
        self.noise = PatternRouter_two(self.d_llm, 1, self.d_ff, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, cycle_numbers, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, 3, fixed_length_of_curve]
            DKP_embeddings: [B, d_llm]
        '''
        B, L = cycle_curve_data.shape[0], cycle_curve_data[1]
        clean_logits = self.gate(DKP_embeddings, cycle_numbers)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings, cycle_numbers)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        _, indices = torch.topk(logits, self.top_k, dim=1)
        # Create a mask where only the top-K values will be kept
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = self.general_expert(cycle_curve_data) + total_outs

        aug_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

        return final_out, aug_loss
    
class IntraCycleMoELayer(nn.Module):
    def __init__(self, configs):
        super(IntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.num_experts = configs.num_experts
        self.activation = configs.activation
        self.top_k = configs.topK
        self.experts = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.general_expert = MLPBlock(self.d_model, self.d_ff, self.drop_rate, self.activation)
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouter_two(self.d_llm, 1, self.d_ff, self.num_experts)
        self.noise = PatternRouter_two(self.d_llm, 1, self.d_ff, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, cycle_numbers, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            cycle_numbers: [B, 1]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings, cycle_numbers)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings, cycle_numbers)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        _, indices = torch.topk(logits, self.top_k, dim=1)
        # Create a mask where only the top-K values will be kept
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = self.general_expert(cycle_curve_data) + total_outs

        aug_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

        return final_out, aug_loss

class FlattenInterCycleMoELayer(nn.Module):
    def __init__(self, configs):
        super(FlattenInterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.num_experts = configs.num_experts
        self.activation = configs.activation
        self.top_k = configs.topK
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model)) for _ in range(self.num_experts)])
        self.general_expert = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model))
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouter_three(self.d_llm, 1, self.early_cycle_threshold*self.d_model, self.d_ff, self.num_experts)
        self.noise = PatternRouter_three(self.d_llm, 1, self.early_cycle_threshold*self.d_model, self.d_ff, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, cycle_numbers, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            cycle_numbers: [B, 1]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings, cycle_numbers, cycle_curve_data.reshape(B, -1))
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings, cycle_numbers.reshape(B, -1))
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        _, indices = torch.topk(logits, self.top_k, dim=1)
        # Create a mask where only the top-K values will be kept
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = self.general_expert(cycle_curve_data) + total_outs

        aug_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

        return final_out, aug_loss
    
class InterCycleMoELayer(nn.Module):
    def __init__(self, configs):
        super(InterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.num_experts = configs.num_experts
        self.activation = configs.activation
        self.top_k = configs.topK
        self.experts = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.general_expert = MLPBlock(self.d_model, self.d_ff, self.drop_rate, self.activation)
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouter_three(self.d_llm, 1, self.d_model, self.d_ff, self.num_experts)
        self.noise = PatternRouter_three(self.d_llm, 1, self.d_model, self.d_ff, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, cycle_numbers, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, d_model]
            cycle_numbers: [B, 1]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings, cycle_numbers, cycle_curve_data)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings, cycle_numbers, cycle_curve_data)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        _, indices = torch.topk(logits, self.top_k, dim=1)
        # Create a mask where only the top-K values will be kept
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = self.general_expert(cycle_curve_data) + total_outs

        aug_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

        return final_out, aug_loss
    
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


        self.projection = nn.Sequential(nn.Linear(self.d_model, self.d_ff), nn.ReLU(),
                                        nn.Linear(self.d_ff, ec_config.output_num))
        
    
    def forward(self, llm_out, attn_mask):
        '''
        llm_out: [N, L, d_llm]
        llm_attn_mask: [N, L]
        curve_attn_mask: [N, L]
        '''
        out = self.projection(llm_out)
        
        return out, llm_out, llm_out

class Model(BatteryLifeLLM):
    def __init__(self, battery_life_config):
        super(Model, self).__init__(battery_life_config)
        configs = battery_life_config.ec_config.get_configs()
        self.configs = configs
        self.task_name = configs.task_name
        self.d_ff = configs.d_ff
        self.top_k = configs.topK
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
        self.cycle_mean = (configs.seq_len+configs.early_cycle_threshold) / 2
        self.cycle_std = np.std([i for i in range(configs.seq_len, configs.early_cycle_threshold+1)])
        self.num_experts = configs.num_experts
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.d_model = configs.d_model
        self.d_llm = configs.d_llm
        self.tokenizer.padding_side = 'right' # set the padding side
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.flattenIntraCycleLayer = FlattenIntraCycleMoELayer(configs)
        self.intraCycleLayers = nn.ModuleList([IntraCycleMoELayer(configs) for _ in range(configs.e_layers)])
        self.flattenInterCycleLayer = FlattenInterCycleMoELayer(configs)
        self.interCycleLayers = nn.ModuleList([InterCycleMoELayer(configs) for _ in range(configs.d_layers)])
        self.regression_head = OutputHead(battery_life_config.ec_config)
        
    def forward(self, cycle_curve_data, curve_attn_mask, input_ids: torch.LongTensor = None,
                end_input_ids: torch.LongTensor = None,
                end_attn_mask: torch.LongTensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                label_input_ids: torch.LongTensor = None,
                label_attention_mask: torch.LongTensor = None,
                label_prompt_embedding: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                DKP_embeddings: Optional[torch.FloatTensor] = None,
                cluster_labels: Optional[torch.LongTensor] = None
                ):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]
        '''
        # process the charge&discharge data
        B, L, num_var, fixed_len = cycle_curve_data.shape[0], cycle_curve_data.shape[1], cycle_curve_data.shape[2], cycle_curve_data.shape[3]

        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        cycle_curve_data, curve_attn_mask = cycle_curve_data.to(torch.bfloat16), curve_attn_mask.to(torch.bfloat16)
        DKP_embeddings = DKP_embeddings.to(torch.bfloat16)
        cycle_numbers = torch.sum(curve_attn_mask, dim=1).unsqueeze(-1) # [B, 1]
        cycle_numbers = (cycle_numbers - self.cycle_mean) / self.cycle_std

        total_aug_loss = 0
        total_aug_count = 0
        out, aug_loss = self.flattenIntraCycleLayer(cycle_curve_data, cycle_numbers, DKP_embeddings) # [B, L, d_model]
        total_aug_loss += aug_loss
        total_aug_count += 1

        for i, expert in enumerate(self.intraCycleLayers):
            out, aug_loss = expert(out, cycle_numbers, DKP_embeddings)
            total_aug_loss += aug_loss
            total_aug_count += 1
        
        out, aug_loss = self.flattenInterCycleLayer(out, cycle_numbers, DKP_embeddings)
        total_aug_loss += aug_loss
        total_aug_count += 1

        for i, expert in enumerate(self.interCycleLayers):
            out, aug_loss = expert(out, cycle_numbers, DKP_embeddings)
            total_aug_loss += aug_loss
            total_aug_count += 1

        preds, llm_out, feature_llm_out = self.regression_head(out, attention_mask)

        preds = preds.float()
        llm_out = llm_out.float()


        label_preds, label_llm_out, label_feature_llm_out = None, None, None

        return preds, None, llm_out, feature_llm_out, label_preds, label_feature_llm_out, total_aug_loss / total_aug_count

    def create_causal_mask(self, B, seq_len):
        '''
        return:
            casual mask: [B, L, L]. 0 indicates masked.
        '''
        # Create a lower triangular matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask
