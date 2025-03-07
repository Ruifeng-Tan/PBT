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
from layers.distributional_router_encoder import DistributionRouter
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
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
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
    
class BatteryExpert(nn.Module):
    '''
    This is the prompt learner that is used to learn prompts that are conditioned on the cycle data
    '''
    def __init__(self, configs):
        super(BatteryExpert, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length # There two summary tokens
        self.drop_rate = configs.dropout
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.patch_len = 10
        self.stride = 10
        self.e_layers = configs.e_layers
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length*3, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])
        
        self.flatten_head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.d_model*configs.early_cycle_threshold, self.d_model))

    
    def forward(self, cycle_curve_data, curve_attn_mask, DKP_embeddings):
        '''
        params:
            cycle_curve_data: [B, L, 3, fixed_length_of_curve]
            curve_attn_mask: [B, L]
            DKP_embeddings: [B, d_llm]
        '''
        DKP_embeddings = DKP_embeddings.unsqueeze(1) # [B, 1, d_llm]
        DKP_attn_mask = torch.ones_like(DKP_embeddings[:,:,0]).long() # [B, 1]
        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, L, fixed_len * num_var]
        cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, L, d_model]

        # Add positional encoding
        whole_cycle_data = cycle_curve_data
        whole_cycle_data = self.flatten_head(whole_cycle_data)
        whole_cycle_data = whole_cycle_data.unsqueeze(-1)  # [B, d_model, 1]
        # curve_attn_mask = torch.cat([DKP_attn_mask, curve_attn_mask], dim=1)

        return whole_cycle_data


class LSTMHead(nn.Module):
    def __init__(self, ec_config):
        super(LSTMHead, self).__init__()
        ec_config = ec_config.get_configs()
        self.d_llm = ec_config.d_llm
        self.d_ff = ec_config.d_ff
        self.d_model = ec_config.d_model
        self.early_cycle_threshold = ec_config.early_cycle_threshold
        self.drop_rate = ec_config.dropout
        self.n_heads = ec_config.n_heads


        self.projection = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                        nn.ReLU(), nn.Linear(self.d_ff, ec_config.output_num))
        
    
    def forward(self, llm_out, attn_mask):
        '''
        llm_out: [N, L, d_llm]
        llm_attn_mask: [N, L]
        curve_attn_mask: [N, L]
        '''
        out = self.projection(llm_out)
        
        return out, llm_out, llm_out
    
class RegressionHead(nn.Module):
    def __init__(self, ec_config):
        super(RegressionHead, self).__init__()
        ec_config = ec_config.get_configs()
        self.d_llm = ec_config.d_llm
        self.d_ff = ec_config.d_ff
        self.early_cycle_threshold = ec_config.early_cycle_threshold
        self.drop_rate = ec_config.dropout
        self.n_heads = ec_config.n_heads
        self.dropout = nn.Dropout(self.drop_rate)
        # if not ec_config.use_DG:
        #     self.projection = nn.Linear(self.d_llm, ec_config.output_num, dtype=torch.bfloat16)
        # else:
        #     # non-linear head must be used to regress the cycle life from the learned representation
        #     self.projection = nn.Sequential(nn.Linear(self.d_llm, self.d_ff, dtype=torch.bfloat16), nn.GELU(),
        #                                     nn.Linear(self.d_ff, ec_config.output_num, dtype=torch.bfloat16))
        # self.translate_layer = nn.Sequential(nn.Linear(self.d_llm, self.d_ff, dtype=torch.bfloat16), nn.GELU(),
        #                                      nn.Linear(self.d_ff, self.d_llm, dtype=torch.bfloat16))
        self.projection = nn.Linear(self.d_llm, ec_config.output_num)
        self.projection_life_class = nn.Linear(self.d_llm, ec_config.class_num)
        
    
    def forward(self, llm_out):
        '''
        llm_out: [N, L, d_llm]
        llm_attn_mask: [N, L]
        curve_attn_mask: [N, L]
        '''
        
        llm_out = llm_out[:,-1,:]
        llm_out = self.dropout(llm_out)
        # feature_llm_out = self.translate_layer(llm_out)
        out = self.projection(llm_out)
        out_life_class = self.projection_life_class(llm_out)
        out_life_class = F.softmax(out_life_class, dim=-1)
        
        return out, out_life_class, llm_out, llm_out
        
    
class PromptTuning(nn.Module):
    def __init__(self, d_llm, drop_rate, d_ff, token_num=30):
        super(PromptTuning, self).__init__()
        self.prompts = nn.Embedding(token_num, d_llm)
        self.token_num = token_num
        self.d_llm = d_llm
        self.d_ff = d_ff
        # self.LSTM = nn.LSTM(d_llm, bidirectional=True, hidden_size=d_llm//2, 
        #                     num_layers=2, batch_first=True, dropout=drop_rate)
        self.mlp_head = nn.Sequential(nn.Linear(self.d_llm, self.d_ff),
                                      nn.ReLU(),
                                      nn.Linear(self.d_ff, self.d_llm))
    
    def forward(self, x):
        '''
        x: [B, *]
        '''
        B = x.shape[0]
        indices = torch.LongTensor(list(range(self.token_num))).to(x.device) # [token_num]
        indices = indices.unsqueeze(0).expand(B, -1) # [B, token_num]
        prompt_embed = self.prompts(indices) # [B, token_num, d_llm]
        # prompt_embed, (_,_) = self.LSTM(prompt_embed) # [B, token_num, d_llm]
        prompt_embed = self.mlp_head(prompt_embed)
        attn_mask = torch.ones_like(prompt_embed[:,:,0])
        return prompt_embed, attn_mask

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
        self.tokenizer.padding_side = 'right' # set the padding side
        self.d_llm = configs.d_llm
        self.topK = configs.topK
        self.softplus = nn.Softplus()
        self.noisy_gating = configs.noisy_gating
        self.gate  = DistributionRouter(configs)
        self.noise = DistributionRouter(configs)
        self.P_token_num = configs.P_token_num
        self.experts = nn.ModuleList([BatteryExpert(configs) for _ in range(configs.num_experts)])
        # self.end_prompt_L = 4
        self.regression_head = LSTMHead(battery_life_config.ec_config)
        
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

        clean_logits = self.gate(DKP_embeddings)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.noise(DKP_embeddings)
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
        mask.scatter_(1, indices, 1)
        # Set all values that are not top-K to zero
        logits = logits * mask.float()

        logits = F.softmax(logits, dim=1)
        logits = logits.unsqueeze(1) # [B, 1, num_experts]
        logits = logits.to(torch.bfloat16)
        
        total_dec_outs = []
        for i, expert in enumerate(self.experts):
            dec_out = expert(cycle_curve_data, curve_attn_mask, DKP_embeddings) # [N, d_llm, 1]
            total_dec_outs.append(dec_out)
        total_dec_outs = torch.cat(total_dec_outs, dim=-1) # [B, d_llm, num_experts]
        dec_out = torch.sum(total_dec_outs * logits, dim=-1) # [B, L, d_llm]
        preds, llm_out, feature_llm_out = self.regression_head(dec_out, attention_mask)

        preds = preds.float()
        llm_out = llm_out.float()


        label_preds, label_llm_out, label_feature_llm_out = None, None, None

        return preds, None, llm_out, feature_llm_out, label_preds, label_feature_llm_out, label_llm_out

    def create_causal_mask(self, B, seq_len):
        '''
        return:
            casual mask: [B, L, L]. 0 indicates masked.
        '''
        # Create a lower triangular matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask
