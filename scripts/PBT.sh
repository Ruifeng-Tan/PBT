model_name=PBT
dataset=MIX_75p # MIX_all, MIX_all42, MIX_all2024
seed=2024 # 2021, 42, 2024
train_epochs=100
accumulation_steps=1
cl_epoches=0
least_epochs=80
warm_up_epoches=0
seq_len=1
early_cycle_threshold=100
learning_rate=2.5e-5
lradj_factor=0.5

# loader
num_domains=32

down_sample_ratio=0.75
llm_choice=Llama # [Llama, Qwen3_8B, Qwen3_0.6B]
d_llm=4096 # Qwen3_8B: 4096, Qwen3_0.6B: 1024 
pca_path=/data/trf/python_works/BatteryLife/dataset/MIX_large_pca_0.99_$llm_choice.pkl

tune_layers=4
num_process=2
batch_size=128

master_port=25257

temperature=1.0
aug_w=1.0
n_heads=8
e_layers=2
d_layers=5

d_model=128
d_ff=32
min_d_ff=64
gate_d_ff=512 # 29
dk_factor=15


dropout=0.05
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
wd=0.01
embed=Cycle
P_token_num=4
activation=gelu

# router / gating
num_views=4
num_general_experts=5
num_experts=20
cathode_experts=7 # 4 . 7
temperature_experts=10 # 10
format_experts=6 # 6
anode_experts=6 # 6
topK=-1
cycle_topK=2
importance_weight=0.1
LLM_path=/data/LLMs/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 # /data/LLMs/Meta-Llama-3.1-8B-hf
checkpoints=/data/tmpf # the save path of checkpoints
data=Dataset_PBT
root_path=/data/trf/python_works/BatteryLife/dataset
comment='a' 


# /data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --num_process $num_process \
  --lradj_factor $lradj_factor \
  --task_name battery_life_prediction \
  --data $data \
  --importance_weight $importance_weight \
  --is_training 1 \
  --root_path $root_path \
  --num_experts $num_experts \
  --topK $topK \
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
  --train_epochs $train_epochs \
  --model_comment $comment \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 16 \
  --d_llm $d_llm \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --num_views $num_views \
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
  --cathode_experts $cathode_experts \
  --temperature_experts $temperature_experts \
  --anode_experts $anode_experts \
  --format_experts $format_experts \
  --num_general_experts $num_general_experts \
  --top_p $top_p \
  --gamma $gamma \
  --warm_up_epoches $warm_up_epoches \
  --num_domains $num_domains \
  --temperature $temperature \
  --aug_w $aug_w \
  --down_sample_ratio $down_sample_ratio \
  --pca_path $pca_path \
  --gate_d_ff $gate_d_ff \
  --accumulation_steps $accumulation_steps \
  --llm_choice $llm_choice \
  --cl_epoches $cl_epoches \
  --dk_factor $dk_factor \
  --min_d_ff $min_d_ff \
  --use_dff_scale 