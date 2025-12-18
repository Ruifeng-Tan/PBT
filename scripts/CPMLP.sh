model_name=CPMLP
dataset=RWTH # MIX_all, MIX_all42, MIX_all2024
seed=42 # 2021, 42, 2024
train_epochs=100
warm_up_epoches=0
early_cycle_threshold=100
learning_rate=0.000025
lradj_factor=0.5

# loader
num_domains=32
aug_w=1.0
temperature=1.0
down_sample_ratio=0.75


llm_layers=32 # 70B-80 layers, 8B-32 layers, 3B-28 layers
tune_layers=4
master_port=25263

num_process=2
batch_size=128
n_heads=8
seq_len=1
accumulation_steps=1

e_layers=6
d_layers=6

bottleneck_factor=16 # scale down factor for the bottleneck layer
d_model=128
d_ff=64

d_llm=4096 #70B - 8192;  8B-4096; 3B-3072
dropout=0.25
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant

# top-p
top_p=0.5

# Guidance
gamma=1.0

loss=MSE
# fine-grained pathcing
patch_len=10
stride=10
wd=0.0
least_epochs=10
embed=Cycle
P_token_num=4
activation=relu

# router / gating
LLM_path=/data/LLMs/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 # /data/LLMs/Meta-Llama-3.1-8B-hf
checkpoints=/data/LLMs/checkpoints # the save path of checkpoints
data=Dataset_PBT
root_path=/data/trf/python_works/BatteryLife/dataset
comment='50to1' # Llama2


# /data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --num_process $num_process \
  --lradj_factor $lradj_factor \
  --task_name battery_life_prediction \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id TunePara \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --seed $seed\
  --embed $embed \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 16 \
  --d_llm $d_llm \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --LLM_path $LLM_path \
  --patch_len $patch_len \
  --stride $stride \
  --wd $wd \
  --least_epochs $least_epochs \
  --activation $activation \
  --top_p $top_p \
  --gamma $gamma \
  --warm_up_epoches $warm_up_epoches 

  

  
