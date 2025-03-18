'''
和BatteryMoE_Seek_Connect相同，只不过在gating的时候dynamically select activated number of experts
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
from layers.Embed import PositionalEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, RMSEncoderLayer, MyEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
from layers.Embed import TokenEmbedding, DataEmbedding
from layers.fusion import GatedFusion
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
transformers.logging.set_verbosity_error()

def top_p_mask(logits, p=0.9):
    """
    Creates a mask tensor with the same shape as logits where selected indices are set to 1 and unselected indices are set to 0.

    Parameters:
    - logits: A tensor of shape [B, N] where B is batch size and N is the number of logits.
    - p: The cumulative probability threshold.

    Returns:
    - mask: A tensor of shape [B, N] with 1s for selected indices and 0s for unselected indices.
    """
    # Calculate probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)

    # Sort probabilities and corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask to select indices where cumulative probability is less than p
    mask = cumulative_probs < p

    # Ensure we always select at least one element
    # Find the first index where cumulative probability is greater than p and set it to True
    mask[:, 0] = True

    # Initialize a mask tensor with zeros
    output_mask = torch.zeros_like(logits, dtype=torch.int)

    # Use the mask to set the selected indices in the output mask tensor to 1
    output_mask.scatter_(1, sorted_indices, mask.int())

    return output_mask

def compute_CL_loss(total_expert_outs, total_cluster_centers, tau, is_intra=True):
    '''
    This is used to compute the contrastive learning loss in the intra-cycle/inter-cycle modeling
    params:
        total_expert_outs: a list of tensors [expert_batch_size, L, d_model] if is_intra else [expert_batch_size, d_model]
        total_cluster_centers: [selected_experts, d_model]
        tau: the temperature in contrastive learning
        is_intra: bool. True for intra-cycle modeling, False for inter-cycle modeling
    return:
        cl_loss: scalar. The contrastive learning loss
    '''
    total_cluster_centers = torch.stack(total_cluster_centers, dim=0)
    norm_total_cluster_centers = F.normalize(total_cluster_centers, p=2, dim=-1) # [selected_experts, d_model]
    selected_experts = len(total_expert_outs)
    cl_loss = 0

    for expert_index, expert_outs in enumerate(total_expert_outs):
        if is_intra:
            expert_outs = expert_outs.reshape(-1, expert_outs.shape[-1]) # [expert_batch_size*L, d_model]
        norm_expert_outs = F.normalize(expert_outs, p=2, dim=-1) # [expert_batch_size*L, d_model] if is_intra else [expert_batch_size, d_model]
        sim = torch.mm(norm_expert_outs, norm_total_cluster_centers.transpose(0, 1))  # [expert_batch_size*L, selected_experts]
        sim = torch.exp(sim / tau)
        # print(sim.shape, len(total_expert_outs))
        positive_logits = sim[:, expert_index] # [expert_batch_size*L] the logits of positive samples
        negative_logitis = torch.sum(sim, dim=1) - positive_logits # [expert_batch_size*L] the logitis of negative samples
        cl_loss += -torch.mean(torch.log(positive_logits/negative_logitis)) # scalar
        
    cl_loss /= selected_experts
    return cl_loss    

class MLPBlockSwishGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlockSwishGLU, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.act_linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.Sigmoid()
        self.act_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)

        # SwishGLU
        out = out * self.act(self.act_linear1(out)) * self.act_linear2(out)

        out = self.dropout(out)
        out = self.out_linear(out)
        return out

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.act = nn.ReLU() if activation=='relu' else nn.GELU(approximate='tanh')
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        return self.dropout(out)
    
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
        self.tau = configs.tau
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        self.general_experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_general_experts)])

        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouterMLP(self.d_llm, self.num_experts)
        self.noise = PatternRouterMLP(self.d_llm, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, 3, fixed_length_of_curve]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        mask = top_p_mask(logits, p=self.top_k/self.num_experts) # [B, num_experts]
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        total_cluster_centers = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, L, d_model]
            cluster_center = torch.mean(out.reshape(-1, out.shape[-1]), dim=0) # [d_model]  
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)
                total_cluster_centers.append(cluster_center)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]

        final_out = total_outs
        for i in range(self.num_general_experts):
            final_out = self.general_experts[i](cycle_curve_data) + final_out

        aug_loss = 0
        cl_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

            # Compute the contrastive learning loss
            cl_loss = compute_CL_loss(total_expert_outs, total_cluster_centers, self.tau, is_intra=True)

        return final_out, aug_loss, cl_loss
    
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
        self.tau = configs.tau
        self.experts = nn.ModuleList([MLPBlockSwishGLU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        self.general_experts = nn.ModuleList([MLPBlockSwishGLU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)])
        self.ln = nn.LayerNorm(self.d_model)
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouterMLP(self.d_llm, self.num_experts)
        self.noise = PatternRouterMLP(self.d_llm, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        mask = top_p_mask(logits, p=self.top_k/self.num_experts) # [B, num_experts]
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        total_cluster_centers = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, L, d_model]
            cluster_center = torch.mean(out.reshape(-1, out.shape[-1]), dim=0) # [d_model]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)
                total_cluster_centers.append(cluster_center)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs
        for i in range(self.num_general_experts):
            final_out = self.general_experts[i](cycle_curve_data) + final_out
        final_out = self.ln(final_out + cycle_curve_data) # add & norm

        aug_loss = 0
        cl_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

            # Compute the contrastive learning loss
            cl_loss = compute_CL_loss(total_expert_outs, total_cluster_centers, self.tau, is_intra=True)

        return final_out, aug_loss, cl_loss

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
        self.num_experts = configs.d_num_experts
        self.activation = configs.activation
        self.top_k = configs.topK
        self.tau = configs.tau
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model)) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        self.general_experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model)) for _ in range(self.num_general_experts)])
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouterMLP(self.d_llm, self.num_experts)
        self.noise = PatternRouterMLP(self.d_llm, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        mask = top_p_mask(logits, p=self.top_k/self.num_experts) # [B, num_experts]
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        total_cluster_centers = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_model]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)
                total_cluster_centers.append(torch.mean(out, dim=0))

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, d_model]
        final_out = total_outs
        for i in range(self.num_general_experts):
            final_out = self.general_experts[i](cycle_curve_data) + final_out

        aug_loss = 0
        cl_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

            # Compute the contrastive learning loss
            cl_loss = compute_CL_loss(total_expert_outs, total_cluster_centers, self.tau, is_intra=False)

        return final_out, aug_loss, cl_loss
    
class InterCycleMoELayer(nn.Module):
    def __init__(self, configs):
        super(InterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.num_experts = configs.d_num_experts
        self.activation = configs.activation
        self.top_k = configs.topK
        self.tau = configs.tau
        self.experts = nn.ModuleList([MLPBlockSwishGLU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        self.general_experts = nn.ModuleList([MLPBlockSwishGLU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)])
        self.ln = nn.LayerNorm(self.d_model)
        self.noisy_gating = configs.noisy_gating
        self.gate  = PatternRouterMLP(self.d_llm, self.num_experts)
        self.noise = PatternRouterMLP(self.d_llm, self.num_experts)
        self.softplus = nn.Softplus()
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, d_model]
            DKP_embeddings: [B, d_llm]
        '''
        B = cycle_curve_data.shape[0]
        clean_logits = self.gate(DKP_embeddings)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings)
            noise_stddev = ((self.softplus(raw_noise_stddev) + 1e-2))
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits # [B, num_experts]

        mask = top_p_mask(logits, p=self.top_k/self.num_experts) # [B, num_experts]
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        total_cluster_centers = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)
                total_cluster_centers.append(torch.mean(out, dim=0))

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs
        for i in range(self.num_general_experts):
            final_out = self.general_experts[i](cycle_curve_data) + final_out
        final_out = self.ln(final_out + cycle_curve_data) # add & norm

        aug_loss = 0
        cl_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / B # [num_experts]
            aug_loss = torch.mean(expert_logits * expert_sample_count) # [1]

            # Compute the contrastive learning loss
            cl_loss = compute_CL_loss(total_expert_outs, total_cluster_centers, self.tau, is_intra=False)

        return final_out, aug_loss, cl_loss
    
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
        
    
    def forward(self, llm_out, attn_mask):
        '''
        llm_out: [N, L, d_llm]
        llm_attn_mask: [N, L]
        curve_attn_mask: [N, L]
        '''
        out = self.projection(llm_out)
        
        return out, llm_out, llm_out

class Model(BatteryLifeLLM):
    '''
    The load balancing loss is from the paper "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity".
    '''
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
            curve_attn_mask: [B, L]. 0 indicates masked
        '''
        # process the charge&discharge data
        B, L, num_var, fixed_len = cycle_curve_data.shape[0], cycle_curve_data.shape[1], cycle_curve_data.shape[2], cycle_curve_data.shape[3]

        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        cycle_curve_data, curve_attn_mask = cycle_curve_data.to(torch.bfloat16), curve_attn_mask.to(torch.bfloat16)
        DKP_embeddings = DKP_embeddings.to(torch.bfloat16)

        total_aug_loss = 0
        total_cl_loss = 0
        total_aug_count = 0
        out, aug_loss, cl_loss = self.flattenIntraCycleLayer(cycle_curve_data, DKP_embeddings) # [B, L, d_model]
        total_aug_loss += aug_loss
        total_cl_loss += cl_loss
        total_aug_count += 1

        for i, expert in enumerate(self.intraCycleLayers):
            out, aug_loss, cl_loss = expert(out, DKP_embeddings)
            total_aug_loss += aug_loss
            total_cl_loss += cl_loss
            total_aug_count += 1
        
        out, aug_loss, cl_loss = self.flattenInterCycleLayer(out, DKP_embeddings)
        total_aug_loss += aug_loss
        total_cl_loss += cl_loss
        total_aug_count += 1

        for i, expert in enumerate(self.interCycleLayers):
            out, aug_loss, cl_loss = expert(out, DKP_embeddings)
            total_aug_loss += aug_loss
            total_cl_loss += cl_loss
            total_aug_count += 1

        preds, llm_out, feature_llm_out = self.regression_head(out, attention_mask)

        preds = preds.float()
        llm_out = llm_out.float()


        label_preds, label_llm_out, label_feature_llm_out = None, None, None

        return preds, None, llm_out, feature_llm_out, label_preds, label_feature_llm_out, total_aug_loss / total_aug_count, total_cl_loss / total_aug_count

    def create_causal_mask(self, B, seq_len):
        '''
        return:
            casual mask: [B, L, L]. 0 indicates masked.
        '''
        # Create a lower triangular matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask
