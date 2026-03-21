args_path=/home/hwx/python_project/test/BatteryLife/checkpoints/CPTransformer_sl1_lr5e-05_dm128_nh4_el6_dl0_df256_lradjconstant_datasetMIX_large_lossMSE_wd0.0_wlTrue_bs32_s42-CPTransformer/ # the source checkpoints

batch_size=32
finetune_method=FT
finetune_type=CPT2CPT
num_process=1
master_port=24988
finetune_dataset=CALB42 # the target dataset
model_name=CPTransformer
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
accumulation_steps=1
lstm_layers=2
d_model=64
d_ff=128
e_layers=2
adapter_size=128
loss=MSE
seq_len=1
d_layers=2 
dropout=0.1
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
n_heads=8
seed=42

checkpoints=/data/trf/finetune_AT_checkpoints # the save path of checkpoints
data=Dataset_original
root_path=/data/trf/python_works/PBT_BatteryLife/dataset
comment='CPMLP' 
task_name=classification


CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes $num_process --main_process_port $master_port finetune_model.py \
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
  --lstm_layers $lstm_layers \
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
  --finetune_method $finetune_method \
  --warm_up_epoches 0 \
  --finetune_type $finetune_type
