# Dedicated entry for evaluating BatLiNet checkpoints.
# It simply drives the shared evaluate_model.py with --model BatLiNet,
# which reproduces the original BatLiNet evaluation protocol
# (support-set construction identical to training).

args_path=/data/hwx/BLN/BatLiNet_sl1_lr5e-05_dm128_nh8_el4_dl2_df256_lradjconstant_datasetLFP_tdata_CALCE_lossMSE_wd0.0_wlFalse_bs64_s2024-BatLiNet/ # the model you want to evaluate

batch_size=16
num_process=2
master_port=26945
eval_cycle_min=-1 # set eval_cycle_min as 1 and eval_cycle_max as 100 to evaluate all samples
eval_cycle_max=-1
eval_dataset=CALCE
model=BatLiNet
root_path=/data/trf/python_works/BatteryLife/dataset # optional; defaults to the root_path saved in the checkpoint args.json

CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model \
  --root_path $root_path
