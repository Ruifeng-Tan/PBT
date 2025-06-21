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
from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer, RMSEncoderLayer
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
from layers.MLPs import MLPBlockGELU
transformers.logging.set_verbosity_error() 

class MultiViewLayer(nn.Module):
    def __init__(self, gate_input_dim, num_experts, view_experts, norm_layer, general_experts, ion_experts, use_connection, use_norm=True):
        super(MultiViewLayer, self).__init__()
        self.num_views = len(view_experts)
        self.num_general_experts = len(general_experts)
        self.ion_experts = ion_experts # when multiple ion types are available in the training set, we have ion experts for different ion type
        self.expert_gate = nn.Linear(gate_input_dim, num_experts, bias=False)
        
        self.view_experts = view_experts
        self.general_experts = general_experts
        self.use_connection = use_connection
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = norm_layer

    def forward(self, x, gate_input, total_masks, ion_type_masks, use_view_experts):
        '''
        x: [N, *, in_dim]
        gate_input: [B, gate_input_dim]
        total_masks: [num_view, num_experts for each view expert]
        ion_type_masks: [B, ion_expert_num]. 1 indicates activated
        '''
        x = self.norm(x) if self.use_norm else x # pre norm
        B = x.shape[0]
        total_guide_loss = 0
        total_LB_loss = 0
        final_out = 0
        total_logits = self.expert_gate(gate_input) # [B, num_experts]
       
        if use_view_experts:
            for i, view_expert in enumerate(self.view_experts):
                out, guide_loss, LB_loss = view_expert(x, total_logits, total_masks[i])
                final_out = final_out + out
                total_guide_loss += guide_loss
                total_LB_loss += LB_loss

        for i in range(len(self.general_experts)):
            final_out = self.general_experts[i](x) + final_out # add the general experts

        if len(self.ion_experts) != 0:
            total_ion_outs = [] # each element is [B, 1, *, D]
            for i in range(len(self.ion_experts)):
                ion_out = self.ion_experts[i](x)
                total_ion_outs.append(ion_out.unsqueeze(1))

            total_ion_outs = torch.cat(total_ion_outs, dim=1) # [B, ion_expert_num, *, D]
            if total_ion_outs.dim() == 4:
                ion_type_masks = ion_type_masks.reshape(B, ion_type_masks.shape[1], 1, 1)
            else:
                ion_type_masks = ion_type_masks.reshape(B, ion_type_masks.shape[1], 1)
            total_ion_outs = torch.sum(total_ion_outs * ion_type_masks, dim=1)
            final_out = final_out + total_ion_outs

        if self.use_connection:
            final_out = final_out + x # residual connection
        
        # final_out = self.norm(final_out) if self.use_norm else final_out # pre norm
        return final_out, total_guide_loss / self.num_views, total_LB_loss / self.num_views
    


class MultiViewTransformerLayer(nn.Module):
    def __init__(self, gate_input_dim, num_experts, d_model, n_heads, view_experts, general_experts, ion_experts, drop_rate):
        super(MultiViewTransformerLayer, self).__init__()
        self.num_views = len(view_experts)
        self.num_general_experts = len(general_experts)
        self.ion_experts = ion_experts # when multiple ion types are available in the training set, we have ion experts for different ion type
        self.expert_gate = nn.Linear(gate_input_dim, num_experts)

        self.attention = AttentionLayer(FullAttention(True, 1, attention_dropout=drop_rate,
                            output_attention=False), d_model, n_heads)
        self.dropout = nn.Dropout(drop_rate)
        self.view_experts = view_experts
        self.general_experts = general_experts

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, gate_input, total_masks, attn_mask, ion_type_masks, use_view_experts):
        '''
        x: [N, *, in_dim]
        gate_input: [B, gate_input_dim]
        total_masks: [num_view, num_experts for each view expert]
        attn_mask: [B, 1, L, L]
        '''
        B = x.shape[0]
        x = self.norm1(x)
        # casual masked self-attention
        new_x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=None, delta=None
        )
        x = x + self.dropout(new_x) # residual connection 
        x = self.norm2(x)

        # MoE FFN
        total_guide_loss = 0
        total_LB_loss = 0
        final_out = 0
        total_logits = self.expert_gate(gate_input) # [B, num_experts]
        if use_view_experts:
            for i, view_expert in enumerate(self.view_experts):
                out, guide_loss, LB_loss = view_expert(x, total_logits, total_masks[i])
                final_out = final_out + out
                total_guide_loss += guide_loss
                total_LB_loss += LB_loss

        for i in range(len(self.general_experts)):
            final_out = self.general_experts[i](x) + final_out # add the general experts

        if len(self.ion_experts) != 0:
            total_ion_outs = [] # each element is [B, 1, *, D]
            for i in range(len(self.ion_experts)):
                ion_out = self.ion_experts[i](x)
                total_ion_outs.append(ion_out.unsqueeze(1))

            total_ion_outs = torch.cat(total_ion_outs, dim=1) # [B, ion_expert_num, *, D]
            if total_ion_outs.dim() == 4:
                ion_type_masks = ion_type_masks.reshape(B, ion_type_masks.shape[1], 1, 1)
            else:
                ion_type_masks = ion_type_masks.reshape(B, ion_type_masks.shape[1], 1)
            total_ion_outs = torch.sum(total_ion_outs * ion_type_masks, dim=1)
            final_out = final_out + total_ion_outs

        final_out = self.dropout(final_out) + x # residual connection 
        # final_out = self.norm2(self.dropout(final_out) + x) # add & norm
        return final_out, total_guide_loss / self.num_views, total_LB_loss / self.num_views
    
class BatteryMoEFlattenIntraCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, d_ff_scale_factor):
        super(BatteryMoEFlattenIntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts # 4 types of cathodes in the training data
        self.top_k = configs.topK
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(self.charge_discharge_length*3, self.d_model)) for i in range(self.num_experts)])
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
        logits = logits * mask

        
        if self.top_k > 0:
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
            if len(MOE_indicies[i])>=1:
                out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
                total_outs.append(out)
                total_expert_outs.append(out)


        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]

        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out

        guide_loss = 0 # guide the model to give larger weight to the correct cathode expert
        LB_loss = 0
        if self.training:
            # Guidance loss
            # masked_raw_logits = raw_logits * mask
            # sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            # guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)

            # new Guidance loss
            active_logits = raw_logits * mask
            inactive_logits = raw_logits * (1-mask)
            guide_loss = -torch.mean(torch.log(torch.sum(active_logits, dim=1).exp() / torch.sum(inactive_logits, dim=1).exp()))

            # Compute the load balancing loss
            entropy = - logits * torch.log(logits + self.eps) # [B, num_experts]
            entropy = entropy * moe_masks # mask the inactive logits
            entropy_loss = torch.sum(entropy, dim=1) # [B]. The entropy of the logits
            LB_loss = - torch.mean(entropy_loss) # [1]
        
        return final_out, guide_loss, LB_loss

class BatteryMoEIntraCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, d_ff_scale_factor):
        super(BatteryMoEIntraCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model  
        self.num_experts = num_experts # 4 types of cathodes in the training data
        self.top_k = configs.topK
        self.use_dff_scale = configs.use_dff_scale
        self.activation = configs.activation
        self.min_d_ff = configs.min_d_ff
        if self.use_dff_scale:
            self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, max([math.ceil(self.d_ff * d_ff_scale_factor[i]), self.min_d_ff]), self.drop_rate, self.activation) for i in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for i in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
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

        if self.top_k > 0:
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
            if len(MOE_indicies[i])>=1:
                out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
                total_outs.append(out)
                total_expert_outs.append(out)


        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs
        # for i in range(self.num_general_experts):
        #     final_out = self.general_experts[i](cycle_curve_data) + final_out
        # final_out = self.ln(final_out + cycle_curve_data) # add & norm

        guide_loss = 0
        LB_loss = 0
        if self.training:
            # Guidance loss
            # masked_raw_logits = raw_logits * mask
            # sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            # guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)

            # new Guidance loss
            active_logits = raw_logits * mask
            inactive_logits = raw_logits * (1-mask)
            guide_loss = -torch.mean(torch.log(torch.sum(active_logits, dim=1).exp() / torch.sum(inactive_logits, dim=1).exp()))
            
            # Compute the load balancing loss
            entropy = - logits * torch.log(logits + self.eps) # [B, num_experts]
            entropy = entropy * moe_masks # mask the inactive logits
            entropy_loss = torch.sum(entropy, dim=1) # [B]. The entropy of the logits
            LB_loss = - torch.mean(entropy_loss) # [1]

        return final_out, guide_loss, LB_loss
  
class BatteryMoEInterCycleMoELayer(nn.Module):
    def __init__(self, configs, num_experts, d_ff_scale_factor):
        super(BatteryMoEInterCycleMoELayer, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads

        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.top_k = configs.topK
        self.d_model = configs.d_model  
        self.num_experts = num_experts 
        self.activation = configs.activation
        self.min_d_ff = configs.min_d_ff
        self.use_dff_scale = configs.use_dff_scale
        if self.use_dff_scale:
            self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, max([math.ceil(self.d_ff * d_ff_scale_factor[i]), self.min_d_ff]), self.drop_rate, self.activation) for i in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for i in range(self.num_experts)])
        self.num_general_experts = configs.num_general_experts
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

        if self.top_k > 0:
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
            if len(MOE_indicies[i])>=1:
                out = expert(cycle_curve_data[MOE_indicies[i]]) # [expert_batch_size, d_llm]
                total_outs.append(out)
                total_expert_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B, L, d_model]
        final_out = total_outs

        LB_loss = 0
        guide_loss = 0
        if self.training:
            # Guidance loss
            # masked_raw_logits = raw_logits * mask
            # sum_masked_raw_logits = torch.sum(masked_raw_logits) / B
            # guide_loss = (1-sum_masked_raw_logits)*(1-sum_masked_raw_logits)

            # new Guidance loss
            active_logits = raw_logits * mask
            inactive_logits = raw_logits * (1-mask)
            guide_loss = -torch.mean(torch.log(torch.sum(active_logits, dim=1).exp() / torch.sum(inactive_logits, dim=1).exp()))

            # Compute the load balancing loss
            entropy = - logits * torch.log(logits + self.eps) # [B, num_experts]
            entropy = entropy * moe_masks # mask the inactive logits
            entropy_loss = torch.sum(entropy, dim=1) # [B]. The entropy of the logits
            LB_loss = - torch.mean(entropy_loss) # [1]

        return final_out, guide_loss, LB_loss
    
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

        assert self.gate_d_ff >= self.gate_domain_knowledge_neurons, Exception('The gate neurons should be no less than the domain-knowledge neurons')
        self.gate = nn.Sequential(nn.Linear(self.d_llm, self.gate_d_ff, bias=False))
        gate_input_dim = self.gate_d_ff
        self.split_dim = self.d_model // self.num_views
        self.d_ff_scale_factor = configs.d_ff_scale_factor
        
        self.flatten = nn.Flatten(start_dim=2)
        self.flattenIntraCycleLayer = MultiViewLayer(gate_input_dim, self.num_experts,
                                                     nn.ModuleList([BatteryMoEFlattenIntraCycleMoELayer(configs, self.num_experts, self.d_ff_scale_factor)]
                                                                    ),
                                                    norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        nn.Sequential(nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.num_general_experts)
                                                    ]),
                                                    ion_experts=nn.ModuleList([
                                                        nn.Sequential(nn.Linear(self.charge_discharge_length*3, self.d_model)) for _ in range(self.ion_experts)
                                                    ]),
                                                    use_connection=False, use_norm=False)
        
        self.intra_MoE_layers = nn.ModuleList([MultiViewLayer(gate_input_dim, self.num_experts,
                                                     nn.ModuleList([BatteryMoEIntraCycleMoELayer(configs, self.num_experts, self.d_ff_scale_factor)
                                                    ]),
                                                    norm_layer=nn.LayerNorm(self.d_model),
                                                    general_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)
                                                    ]),
                                                    ion_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.ion_experts)
                                                    ]),
                                                    use_connection=True) for _ in range(self.e_layers)])
        
        self.pe = PositionalEmbedding(self.d_model)
        self.inter_MoE_layers = nn.ModuleList([MultiViewTransformerLayer(gate_input_dim, self.num_experts,self.d_model, self.n_heads,
                                                     nn.ModuleList([BatteryMoEInterCycleMoELayer(configs, self.num_experts, self.d_ff_scale_factor),
                                                    ]), 
                                                    general_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)
                                                    ]),
                                                    ion_experts=nn.ModuleList([
                                                        MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.ion_experts)
                                                    ]),
                                                    drop_rate=self.drop_rate
                                                    )
                                             for _ in range(self.d_layers)])
        
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


        total_masks = [combined_masks]

        DKP_embeddings = self.gate(DKP_embeddings) # [B, gate_d_ff]
        DKP_mask = torch.ones_like(DKP_embeddings[:, self.gate_domain_knowledge_neurons:])
        domain_knowledge_ReLU_mask = combined_masks.repeat_interleave(dim=1, repeats=self.dk_factor)
        DKP_mask = torch.cat([DKP_mask, domain_knowledge_ReLU_mask], dim=1)
        DKP_embeddings = F.relu(DKP_embeddings * DKP_mask)
        # if self.gate_d_ff > self.gate_domain_knowledge_neurons:
        #     DKP_embeddings[:, :self.gate_d_ff-self.gate_domain_knowledge_neurons] = F.relu(DKP_embeddings[:, :self.gate_d_ff-self.gate_domain_knowledge_neurons])
        # logits = self.gate(DKP_embeddings)
        # logits = logits.reshape(DKP_embeddings.shape[0], -1, self.num_experts)
  
        logits_index = 0

        total_aug_loss = 0
        total_guide_loss = 0
        total_LB_loss = 0
        total_aug_count = 0

        # cycle_curve_data = self.view_linear(cycle_curve_data) # flatten & linear
        cycle_curve_data = self.flatten(cycle_curve_data)
        out, guide_loss, LB_loss = self.flattenIntraCycleLayer(cycle_curve_data, DKP_embeddings, total_masks, ion_type_masks=ion_type_masks, use_view_experts=use_view_experts) # [B, L, d_model]
        total_guide_loss += guide_loss
        total_LB_loss += LB_loss
        total_aug_count += 1
        logits_index += 1

        for i, intra_MoELayer in enumerate(self.intra_MoE_layers):
            out, guide_loss, LB_loss = intra_MoELayer(out, DKP_embeddings, total_masks, ion_type_masks=ion_type_masks, use_view_experts=use_view_experts) # [B, L, d_model]
            total_guide_loss += guide_loss
            total_LB_loss += LB_loss
            total_aug_count += 1
            logits_index += 1


        # Inter-cycle modelling using Transformer with MoE FFN
        out = out + self.pe(out) # add positional encoding
        attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        attn_mask = torch.repeat_interleave(attn_mask, attn_mask.shape[-1], dim=1) # [B, L, L]
        attn_mask = attn_mask.unsqueeze(1) # [B, 1, L, L]
        attn_mask = attn_mask==0 # set True to mask
        for i, inter_MoELayer in enumerate(self.inter_MoE_layers):
            out, guide_loss, LB_loss = inter_MoELayer(out, DKP_embeddings, total_masks, attn_mask=attn_mask, ion_type_masks=ion_type_masks, use_view_experts=use_view_experts) # [B, L, d_model]
            total_guide_loss += guide_loss
            total_LB_loss += LB_loss
            total_aug_count += 1
            logits_index += 1

        lengths = torch.sum(curve_attn_mask, dim=1).cpu() # [N]
        idx = (torch.as_tensor(lengths, device=out.device, dtype=torch.long) - 1).view(-1, 1).expand(
            len(lengths), out.size(2))
        idx = idx.unsqueeze(1)
        out = out.gather(1, idx).squeeze(1) # [B, D]

        out = self.norm(out)
        preds, embeddings, feature_llm_out = self.regression_head(out)

        preds = preds.float()
        embeddings = embeddings.float()
        return preds[:B], None, embeddings[B:], feature_llm_out, None, None, total_LB_loss / total_aug_count , total_guide_loss / total_aug_count

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
