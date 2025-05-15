from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM, AutoConfig, AutoModel, AutoTokenizer
import torch
from utils.tools import train_model_course, get_parameter_number
LLM_path = '/data/LLMs/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519'
llama_config = AutoConfig.from_pretrained(LLM_path)
if 'Qwen' not in LLM_path:
    tokenizer = AutoTokenizer.from_pretrained(LLM_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(LLM_path, pad_token='<|endoftext|>')
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'cls_token':'[cls_token]'})

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.pad_token)
if not tokenizer.pad_token and 'Qwen' not in LLM_path:
    tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token_id)
pad_token_id = tokenizer.pad_token_id
end_sentences = ['<|begin_of_text|>I love you', '<|begin_of_text|>You are my love dafasdfaw faasdwa asfawasd', '<|begin_of_text|>earhth has you']
res = tokenizer(end_sentences, return_tensors="pt", truncation=True, padding=True, max_length=40)
end_input_ids, end_attention_mask = res['input_ids'], res['attention_mask']
print(end_input_ids.shape)
print(end_input_ids)

# Show the model structure
language_model = AutoModel.from_pretrained(
            LLM_path,
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=llama_config
        )
para_res = get_parameter_number(language_model)
print(para_res)
enc_out = language_model.get_input_embeddings()(torch.LongTensor([pad_token_id,pad_token_id])) 
print(enc_out)
# for n,m in language_model.named_modules():
#     print(n, m)
for n,m in language_model.named_parameters():
    print(n)


