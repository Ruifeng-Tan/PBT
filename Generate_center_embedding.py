from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig, AutoModel, AutoTokenizer
import torch
from utils.tools import train_model_course, get_parameter_number
from typing import Dict
import numpy as np
import json

def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    elif strategy == 'last':
        outputs = outputs[:,-1]
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()

total_embeddings = []
# material_type = 'LFP' # ['LFP', 'Layered_oxide]
LLM_path = '/data/LLMs/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95'
llama_config = AutoConfig.from_pretrained(LLM_path)
if 'Qwen' not in LLM_path:
    tokenizer = AutoTokenizer.from_pretrained(LLM_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(LLM_path, pad_token='<|endoftext|>')
tokenizer.padding_side = 'right'
if not tokenizer.pad_token and 'Qwen' not in LLM_path:
    tokenizer.pad_token = tokenizer.eos_token


sentences = ['The battery cycle life is smaller than 300 cycles.',
             'The battery cycle life ranges from 300 to 1000 cycles.',
             'The battery cycle life ranges from 1000 to 2000 cycles.',
             'The battery cycle life is more than 2000 cycles.']
language_model = AutoModel.from_pretrained(
                LLM_path,
                # 'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=llama_config
            )


inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=1024)
embeddings = language_model(**inputs).last_hidden_state
embeddings = embeddings.detach().cpu().numpy()
embeddings = embeddings[:,-1,:]
print(embeddings.shape)
np.save(f'{LLM_path}/center.npy', embeddings)





