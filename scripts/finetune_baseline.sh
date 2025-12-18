args_path=/data/LLMs/checkpoints/CPMLP_1_Qwen3_8B_1_le10_bs128_lr2.5e-05_dm128_nh8_el6_dl6_df64_mdf32_lradjconstant_RWTH_guideFalse_LBFalse_lossMSE_wd0.0_wlFalse_dr0.25_gdff32_E6_GE2_K2_SFalse_augFalse_augW1.0_tem1.0_wDGFalse_dsr0.75_we0_ffsFalse_seed42-50to1/ # the source checkpoints
batch_size=32
finetune_method=AT_nCP
num_process=2
master_port=25017
finetune_dataset=RWTH # the target dataset
model_name=CPMLP
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
num_process=2
accumulation_steps=1
d_model=128
d_ff=256
e_layers=4
adapter_size=64
loss=MSE

seq_len=1
d_layers=2
dropout=0
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
n_heads=8
seed=2021

checkpoints=/data/trf/checkpoints # the save path of checkpoints
data=Dataset_PBT
root_path=/data/trf/python_works/BatteryLife/dataset
comment='CPMLP' 
task_name=classification


CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes $num_process --main_process_port $master_port finetune_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --finetune_dataset $finetune_dataset \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id CPMLP \
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
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --num_workers 8 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --seed $seed \
  --adapter_size $adapter_size \
  --finetune_method $finetune_method
