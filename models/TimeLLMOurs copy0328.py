from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer
from layers.Embed import PatchEmbeddingTimeLLM, PatchEmbedding_pe, PositionalEmbedding, DataEmbedding_wo_time

import transformers

from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.rul_linear = nn.Linear(nf, 1)

    def forward(self, x):
        x = self.flatten(x)
        trajectory = self.linear(x)
        rul = self.rul_linear(x)
        return trajectory, rul


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        # self.d_llm = 768# minimum
        # self.d_llm = 1024# medimum
        # self.d_llm = 1280# large
        # self.d_llm = 1600# maximum
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 加载llama-2-7B模型参数
        self.d_llm = 4096
        self.llama_config = LlamaConfig.from_pretrained("/home/trf/python_work/llama/llama2-hf-7b")
        # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = configs.llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.llama = LlamaModel.from_pretrained(
            "/home/trf/python_work/llama/llama2-hf-7b",
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=self.llama_config,
            load_in_4bit=True
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            "/home/trf/python_work/llama/llama2-hf-7b/tokenizer.model",
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True
        )
        print(self.llama)

        # 加载 GPT-2 tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        # # model_dir = "/home/trf/python_work/gpt-2/models/124M" # minimum
        # # model_dir = "/home/trf/python_work/gpt-2/models/355M" # medimum
        # model_dir = "/home/trf/python_work/gpt-2/models/774M" # large
        # # model_dir = "/home/trf/python_work/gpt-2/models/1558M" # maximum

        # # 加载 GPT-2 模型
        # self.gpt2 =  GPT2Model.from_pretrained(model_dir,from_tf=True)


        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for i, (name, param) in enumerate(self.llama.named_parameters()):
            # if 'layernorm' in name:
            #     param.requires_grad = True
            # else:
            param.requires_grad = False

        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
        #         param.requires_grad = True
        #     elif 'mlp' in name and configs.mlp == 1:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding_pe(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        #self.word_embeddings = self.llama.get_input_embeddings().weight
        #self.word_embeddings = self.gpt2.get_input_embeddings().weight
        # self.vocab_size = self.word_embeddings.shape[0]
        # self.num_tokens = 1000
        #self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.soh_embed = nn.Linear(configs.seq_len, configs.d_model)
        self.chemical_embed = DataEmbedding_wo_time(configs.chemical_info_in, configs.d_model)
        self.soh_self_attn = nn.MultiheadAttention(configs.d_model, configs.n_heads, batch_first=True)
        self.soh_self_attn_ln = nn.LayerNorm(configs.d_model)
        self.addtional_attn = nn.MultiheadAttention(configs.d_model, configs.n_heads, batch_first=True)
        # self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.ln = nn.LayerNorm(configs.d_model)
        self.timeseries_projection = nn.Linear(configs.d_model, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, rul = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], rul
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_chemical_info = x_enc[:,:,:-1] # [B, L, N_var]
        x_chemical_info = self.chemical_embed(x_chemical_info.to(torch.bfloat16)) # [B,L, D]
        
        x_enc = self.normalize_layers(x_enc, 'norm')

        x_soh = x_enc[:,:,-1:] # the soh trajectory
        B, T, N = x_soh.size()
        
        x_soh = x_soh.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_soh, dim=1)[0]
        max_values = torch.max(x_soh, dim=1)[0]
        medians = torch.median(x_soh, dim=1).values
        means = torch.mean(x_soh, dim=1)
        lags = self.calcute_lags(x_soh)
        trends = x_soh.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(B):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            means_values_str = str(means[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: The state of health is a crucial degradation indicator in the battery long-term usage."
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            # prompt_ = (
            # f"<|endoftext|>Task description: Forecast the next {self.pred_len} state of health (SOH) points given the previous {self.seq_len} SOH points. Each data point is obtained from a cycle."
            # f"Battery specifications: This is a NMC/graphite pouch cell with a nominal capacity of 2.36 Ah. The electrolyte salt is LiPF6 and solvent is EC:EMC (3:7)."
            # f"Operating condition: The battery is 1 C charged to 4.2 V with a constant voltage hold to 10 mA and 1 C discharged to 3.0 V."
            # f"Empirical knowledge: The SOH generally becomes smaller smoothly, and there is about one knee before the SOH is degraded to 0.08. "
            # f"Input statistics: Min value {min_values_str} Max value {max_values_str} Mean value {means_values_str} Median value {median_values_str} Top five lags are {lags_values_str} <|<endoftext>|>"
            # )
            prompt.append(prompt_)

        # x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llama.get_input_embeddings()(prompt.to(x_enc.device)) # (batch, prompt_token, dim)
        #prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x_enc.device))

        #source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        
        x_soh = x_enc[:,:,-1:]    # [1,50,1]
        x_soh = x_soh.permute(0, 2, 1).contiguous()# x_enc:[1,1,50]

        x_soh_patch, n_vars = self.patch_embedding(x_soh.to(torch.bfloat16)) 
        x_soh_patch = x_soh_patch.reshape(B,-1,x_soh_patch.shape[-1]) # x_soh_patch: [1,L1,D]
        x_soh = self.soh_embed(x_soh.to(torch.bfloat16)) # x_soh: [1,1,D]
        x_soh = torch.cat([x_soh_patch, x_soh], dim=1) # [1,L,D]
        
        out, _ = self.soh_self_attn(x_soh, x_soh, x_soh)
        x_soh = out+x_soh
        x_soh = self.soh_self_attn_ln(x_soh)
        # x_soh = x_soh[:,-1:,:]
        
        additional_info, _ = self.addtional_attn(x_soh, x_chemical_info, x_chemical_info)
        enc_out = additional_info + x_soh # [B,L,D]
        enc_out = self.ln(enc_out)
        
        # x_soh = x_enc[:,:,-1:]    # [1,50,1]
        # x_soh = x_soh.permute(0, 2, 1).contiguous()# x_enc:[1,1,50]

        # x_soh, n_vars = self.patch_embedding(x_soh.to(torch.bfloat16))# enc_out: [1,L1,D]
        # enc_out = x_soh

        #enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        enc_out = self.timeseries_projection(enc_out)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        #llama_enc_out, _ = self.cross_modulaity_selfAttn(llama_enc_out, llama_enc_out, llama_enc_out)
        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state
        
        # gpt2_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # dec_out = self.gpt2(inputs_embeds=gpt2_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out, rul = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        rul = rul[:,0,:]
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out, rul

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads) # n_heads=8
        self.d_keys = d_keys or (d_model // n_heads) # n_heads=8
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        
        # print("sourceembed", (source_embedding.size()))
        # print("key_projection", self.key_projection)
        # print("dkeys", self.d_keys)
        # print("nheads", self.n_heads)
        # print("dmodel", self.d_model)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)

        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
