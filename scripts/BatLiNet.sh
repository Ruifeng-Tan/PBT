model_name=BatLiNet
dataset=HNEI
target_datase=HNEI
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
diff_base=5
alpha=0.5
seed=2021
dropout=0.5
batch_size=64

master_port=21152
num_process=2
accumulation_steps=1
d_model=128
d_ff=256
e_layers=4
loss=MSE
seq_len=1
d_layers=2
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
n_heads=8

channels=32
in_channels=6
input_height=20
input_width=1000
max_cycle_index=20

checkpoints=/data/hwx/outs # the save path of checkpoints
data=Dataset_original
root_path=/data/trf/python_works/BatteryLife/dataset
comment='BatLiNet' 
task_name=classification

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu  --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id BatLiNet \
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
  --seed $seed \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --target_dataset $target_datase \
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
  --in_channels $in_channels \
  --channels $channels \
  --input_height $input_height \
  --input_width $input_width \
  --max_cycle_index $max_cycle_index \
  --diff_base $diff_base \
  --alpha $alpha \
