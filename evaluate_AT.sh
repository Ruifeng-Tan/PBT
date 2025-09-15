args_path=/data/hwx/finetune_AT_checkpoints/CPMLP_sl1_lr5e-05_dm128_nh8_el4_dl2_df256_seed2021_datasetCALB_lossMSE_wd0.0_wlFalse_fmAT_as32-HiMLP/
batch_size=16
num_process=2
master_port=26949
eval_cycle_min=1 # set eval_cycle_min or eval_cycle_max smaller than 0 to evaluate all testing samples
eval_cycle_max=-1
eval_dataset=CALB
model=CPMLP
seed=2021
finetune_method=AT
adapter_size=32

accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_BL.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model \
  --seed $seed \
  --finetune_method $finetune_method \
  --adapter_size adapter_size