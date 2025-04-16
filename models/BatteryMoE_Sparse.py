'''
基于BatteryMoE3_horizontal4_MH，只不过不做view split
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
from layers.Transformer_EncDec import BatteryMoEEncoder, BatteryMoEEncoderLayer, Encoder, EncoderLayer, ConvLayer, RMSEncoderLayer
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

class MLPBlockGELU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlockGELU, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.in_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = self.act(out)
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

class MultiViewLayer(nn.Module):
    def __init__(self, num_views, view_experts, norm_layer, general_experts, use_connection):
        super(MultiViewLayer, self).__init__()
        self.num_views = num_views
        
        self.view_experts = view_experts
        self.general_experts = general_experts
        self.use_connection = use_connection
        if self.use_connection:
            self.norm = norm_layer

    def forward(self, x, total_logits, total_masks):
        '''
        x: [N, *, in_dim]
        total_logits: [num_view, num_experts for each view expert]
        total_masks: [num_view, num_experts for each view expert]
        '''
        total_outs = []
        total_guide_loss = 0
        final_out = 0
        for i, view_expert in enumerate(self.view_experts):
            out, guide_loss = view_expert(x, total_logits[i], total_masks[i])
            final_out = final_out + out
            total_guide_loss += guide_loss

        for i in range(len(self.general_experts)):
            final_out = self.general_experts[i](x) + final_out # add the general experts

        if self.use_connection:
            final_out = self.norm(final_out + x) # add & norm
        return final_out, total_guide_loss / self.num_views

class BatteryMoEFlattenIntraCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, topK=None):
        super(BatteryMoEFlattenIntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts # 4 types of cathodes in the training data
        self.top_k = topK if topK is not None else configs.topK
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_experts)])
        # self.num_general_experts = configs.num_general_experts
        # self.general_experts = nn.ModuleList([nn.Linear(in_dim, self.d_model) for _ in range(self.num_general_experts)])
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, logits, moe_masks):
        '''
        params:
            cycle_curve_data: [B, L, 3, fixed_length_of_curve]
            DKP_embeddings: [B, num_experts]
            moe_masks: [B, num_experts]
        '''
        B = cycle_curve_data.shape[0]
        
        mask = torch.where(moe_masks==1, torch.ones_like(logits), torch.zeros_like(logits))
        
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        _, indices = torch.topk(logits, self.top_k, dim=1) # further keep only top-K
        # Create a mask where only the top-K values will be kept
        top_K_mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        top_K_mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = logits * top_K_mask

        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)
        

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, L, d_model]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]

        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out

        guide_loss = 0 # guide the model to give larger weight to the correct cathode expert
        if self.training:
            # Guidance loss
            masked_raw_logits = raw_logits * mask
            sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)

        return final_out, guide_loss

class BatteryMoEIntraCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, topK=None):
        super(BatteryMoEIntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts # 4 types of cathodes in the training data
        self.top_k = topK if topK is not None else configs.topK
        self.activation = configs.activation
        self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        # self.general_experts = nn.ModuleList([MLPBlockGELU(in_dim, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)])
        # self.ln = nn.LayerNorm(self.d_model)
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, logits, moe_masks):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            logits: [B, num_experts]
            moe_masks: [B, num_experts]
        '''
        B = cycle_curve_data.shape[0]


        mask = torch.where(moe_masks==1, torch.ones_like(logits), torch.zeros_like(logits))
        
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        _, indices = torch.topk(logits, self.top_k, dim=1) # further keep only top-K
        # Create a mask where only the top-K values will be kept
        top_K_mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        top_K_mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = logits * top_K_mask

        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, L, d_model]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out
        # final_out = self.ln(final_out + cycle_curve_data) # add & norm

        guide_loss = 0
        if self.training:
            # Guidance loss
            masked_raw_logits = raw_logits * mask
            sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)

        return final_out, guide_loss

class BatteryMoEFlattenInterCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, topK=None):
        super(BatteryMoEFlattenInterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts
        self.activation = configs.activation
        self.top_k = topK if topK is not None else configs.topK
        self.experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.d_model*self.early_cycle_threshold, self.d_model)) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        # self.general_experts = nn.ModuleList([nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_dim*self.early_cycle_threshold, self.d_model)) for _ in range(self.num_general_experts)])
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, logits, moe_masks):
        '''
        params:
            cycle_curve_data: [B, L, d_model]
            logits: [B, num_experts]
            moe_masks: [B, num_experts]
        '''
        B = cycle_curve_data.shape[0]

        mask = torch.where(moe_masks==1, torch.ones_like(logits), torch.zeros_like(logits))
        
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        _, indices = torch.topk(logits, self.top_k, dim=1) # further keep only top-K
        # Create a mask where only the top-K values will be kept
        top_K_mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        top_K_mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = logits * top_K_mask

        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_model]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)


        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, d_model]
        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out

        guide_loss = 0 # guide the model to give larger weight to the correct cathode expert
        if self.training:
            # Guidance loss
            masked_raw_logits = raw_logits * mask
            sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)


        return final_out, guide_loss
    
class BatteryMoEInterCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, topK=None):
        super(BatteryMoEInterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts 
        self.top_k = topK if topK is not None else configs.topK
        self.activation = configs.activation
        self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
        # self.general_experts = nn.ModuleList([MLPBlockGELU(in_dim, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)])
        # self.ln = nn.LayerNorm(self.d_model)
        self.eps = 1e-9

    
    def forward(self, cycle_curve_data, logits, moe_masks):
        '''
        params:
            cycle_curve_data: [B, d_model]
            logits: [B, num_experts]
            moe_masks: [B, num_experts]
        '''
        B = cycle_curve_data.shape[0]

        mask = torch.where(moe_masks==1, torch.ones_like(logits), torch.zeros_like(logits))
        
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        _, indices = torch.topk(logits, self.top_k, dim=1) # further keep only top-K
        # Create a mask where only the top-K values will be kept
        top_K_mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        top_K_mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = logits * top_K_mask

        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out
        # final_out = self.ln(final_out + cycle_curve_data) # add & norm

        aug_loss = 0
        guide_loss = 0
        if self.training:
            # Guidance loss
            masked_raw_logits = raw_logits * mask
            sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)


        return final_out, guide_loss
    
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
        self.num_views = configs.num_views
        self.cathode_split = self.cathode_experts
        self.num_experts = self.cathode_experts + self.temperature_experts + self.format_experts + self.anode_experts
        self.gate = nn.Sequential(nn.Linear(self.d_llm, self.num_experts*(2+self.moe_layers)))
        self.split_dim = self.d_model // self.num_views

        self.flattenIntraCycleLayer = MultiViewLayer(self.num_views, 
                                                     nn.ModuleList([BatteryMoEFlattenIntraCycleMoELayer(configs, self.cathode_experts, 2),
                                                                    BatteryMoEFlattenIntraCycleMoELayer(configs, self.anode_experts, 1),
                                                                    BatteryMoEFlattenIntraCycleMoELayer(configs, self.temperature_experts, 3),
                                                                    BatteryMoEFlattenIntraCycleMoELayer(configs, self.format_experts, 1)],
                                                                    ),
                                                    norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_general_experts)
                                                    ]),
                                                    use_connection=False)
        
        self.intra_MoE_layers = nn.ModuleList([MultiViewLayer(self.num_views, 
                                                     nn.ModuleList([BatteryMoEIntraCycleMoELayer(configs, self.cathode_experts, 2),
                                                                    BatteryMoEIntraCycleMoELayer(configs, self.anode_experts, 1),
                                                                    BatteryMoEIntraCycleMoELayer(configs, self.temperature_experts, 3),
                                                                    BatteryMoEIntraCycleMoELayer(configs, self.format_experts, 1)
                                                    ]),
                                                    norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)
                                                    ]),
                                                    use_connection=True) for _ in range(self.e_layers)])

        self.flattenInterCycleLayer = MultiViewLayer(self.num_views, 
                                                     nn.ModuleList([BatteryMoEFlattenInterCycleMoELayer(configs, self.cathode_experts, 2),
                                                                    BatteryMoEFlattenInterCycleMoELayer(configs, self.anode_experts, 1),
                                                                    BatteryMoEFlattenInterCycleMoELayer(configs, self.temperature_experts, 3),
                                                                    BatteryMoEFlattenInterCycleMoELayer(configs, self.format_experts, 1)
                                                    ]),
                                                    norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model)) for _ in range(self.num_general_experts)
                                                    ]),
                                                    use_connection=False)
        
        self.inter_MoE_layers = nn.ModuleList([MultiViewLayer(self.num_views, 
                                                     nn.ModuleList([BatteryMoEInterCycleMoELayer(configs, self.cathode_experts, 2),
                                                                    BatteryMoEInterCycleMoELayer(configs, self.anode_experts, 1),
                                                                    BatteryMoEInterCycleMoELayer(configs, self.temperature_experts, 3),
                                                                    BatteryMoEInterCycleMoELayer(configs, self.format_experts, 1)
                                                    ]), norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)
                                                    ]),
                                                    use_connection=True
                                                    )
                                             for _ in range(self.d_layers)])
        self.regression_head = OutputHead(battery_life_config.ec_config)


    def forward(self, cycle_curve_data, curve_attn_mask, 
                attention_mask: Optional[torch.Tensor] = None,
                DKP_embeddings: Optional[torch.FloatTensor] = None,
                cathode_masks: Optional[torch.Tensor] = None,
                temperature_masks: Optional[torch.Tensor] = None,
                format_masks: Optional[torch.Tensor] = None,
                anode_masks: Optional[torch.Tensor] = None,
                combined_masks: Optional[torch.Tensor] = None
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
        total_masks = [cathode_masks, anode_masks, temperature_masks, format_masks]

        logits = self.gate(DKP_embeddings)
        logits = logits.reshape(B, -1, self.num_experts)
        cathode_logits = logits[:,:,:self.cathode_experts]
        anode_logits = logits[:,:,self.cathode_experts:self.cathode_experts+self.anode_experts]
        temperature_logits = logits[:,:,self.cathode_experts+self.anode_experts:self.cathode_experts+self.anode_experts+self.temperature_experts]
        format_logits = logits[:,:,self.cathode_experts+self.anode_experts+self.temperature_experts:]
  
        logits_index = 0

        total_aug_loss = 0
        total_guide_loss = 0
        total_aug_count = 0

        # cycle_curve_data = self.view_linear(cycle_curve_data) # flatten & linear
        out, guide_loss = self.flattenIntraCycleLayer(cycle_curve_data, [cathode_logits[:,logits_index],anode_logits[:,logits_index],temperature_logits[:,logits_index],format_logits[:,logits_index]], total_masks) # [B, L, d_model]
        total_guide_loss += guide_loss
        total_aug_count += 1
        logits_index += 1

        for i, intra_MoELayer in enumerate(self.intra_MoE_layers):
            out, guide_loss = intra_MoELayer(out, [cathode_logits[:,logits_index],anode_logits[:,logits_index],temperature_logits[:,logits_index],format_logits[:,logits_index]], total_masks) # [B, L, d_model]
            total_guide_loss += guide_loss
            total_aug_count += 1
            logits_index += 1

        # out = out.reshape(B, -1) # flatten
        out, guide_loss = self.flattenInterCycleLayer(out, [cathode_logits[:,logits_index],anode_logits[:,logits_index],temperature_logits[:,logits_index],format_logits[:,logits_index]], total_masks)
        total_guide_loss += guide_loss
        total_aug_count += 1
        logits_index += 1

        for i, inter_MoELayer in enumerate(self.inter_MoE_layers):
            out, guide_loss = inter_MoELayer(out, [cathode_logits[:,logits_index],anode_logits[:,logits_index],temperature_logits[:,logits_index],format_logits[:,logits_index]], total_masks)
            total_guide_loss += guide_loss
            total_aug_count += 1
            logits_index += 1

        preds, llm_out, feature_llm_out = self.regression_head(out, attention_mask)

        preds = preds.float()
        llm_out = llm_out.float()

        return preds, None, llm_out, feature_llm_out, None, None, total_aug_loss , total_guide_loss / total_aug_count

    def create_causal_mask(self, B, seq_len):
        '''
        return:
            casual mask: [B, L, L]. 0 indicates masked.
        '''
        # Create a lower triangular matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask
