model_name=PBT
topK=-1
finetune_dataset=CALB42 # ZN-coin, ZN-coin42, ZN-coin2024, NAion
batch_size=8
args_path=/data/LLMs/checkpoints/PBT_10_Llama_1_le80_bs128_lr2.5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_MIX_large_guideFalse_LBFalse_lossMSE_wd0.01_wlFalse_dr0.05_gdff512_E5_GE5_K-1_SFalse_augFalse_augW1.0_tem1.0_wDGFalse_dsr0.75_we0_ffsTrue_seed42-100/
master_port=25270
train_epochs=300

seq_len=1
early_cycle_threshold=100
learning_rate=0.000025
warm_up_epoches=0
adapter_size=16
adapter_layers=-1
wd=0.0
dropout=0.0
loss=MSE
lradj_factor=0.5
finetune_method=FT
# loader
num_domains=32
aug_w=1.0
temperature=1.0
down_sample_ratio=0.75


llm_layers=32 # 70B-80 layers, 8B-32 layers, 3B-28 layers


num_process=1

accumulation_steps=1



bottleneck_factor=16 # scale down factor for the bottleneck layer


charge_discharge_length=300
patience=30 # Eearly stopping patience
lradj=constant

# top-p
top_p=0.5

# Guidance
gamma=1.0


# fine-grained pathcing
patch_len=10
stride=10
least_epochs=50
embed=Cycle
P_token_num=4
activation=relu

# router / gating
num_views=4
num_hyper_experts=0
num_condition_experts=0
num_general_experts=5
num_experts=20
cathode_experts=11 # 4 . large: 11. All: 13
temperature_experts=14 # 9.large: 14. All: 18
format_experts=11 # 4. large: 11. All: 13
anode_experts=11 # 2. large: 11. All:13
ion_experts=0
cycle_topK=2
importance_weight=1.0
checkpoints=/data/tmpf # the save path of checkpoints
data=Dataset_BatteryLifeLLM_original
root_path=/data/trf/python_works/BatteryLife/dataset
comment='b' # Llama2


# /data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
# --use_aug \
CUDA_VISIBLE_DEVICES=7 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port finetune_PBT.py \
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
  --embed $embed \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --num_workers 16 \
  --num_views $num_views \
  --patience $patience \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --wd $wd \
  --least_epochs $least_epochs \
  --gamma $gamma \
  --warm_up_epoches $warm_up_epoches \
  --aug_w $aug_w \
  --down_sample_ratio $down_sample_ratio \
  --args_path $args_path \
  --finetune_dataset $finetune_dataset \
  --temperature $temperature \
  --finetune_method $finetune_method \
  --adapter_size $adapter_size \
  --adapter_layers $adapter_layers

  

  
