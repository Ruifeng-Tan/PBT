args_path=/data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB42_dr0.1_FT_seed42-CPMLP/ # the model you want to evaluate

# AT_CALB_CPMLP_checkpoints
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB42_dr0.1_AT_seed42-CPMLP/
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB_dr0.1_AT_seed2021-CPMLP/
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB2024_dr0.1_AT_seed2024-CPMLP/

# AT_CALB_CPTransformer_checkpoints
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el6_dl0_df256_CALB42_dr0.1_AT_seed42-CPTransformer/
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el12_dl0_df256_CALB_dr0.1_AT_seed2021-CPTransformer/
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el12_dl0_df256_CALB2024_dr0.1_AT_seed2024-CPTransformer/

# FT_CALB_CPMLP_checkpoints
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB42_dr0.1_FT_seed42-CPMLP/
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB_dr0.1_FT_seed2021-CPMLP/
# /data/hwx/PBT_transfer/CPMLP_as128_al2_bs8_lr5e-05_dm128_el4_dl2_df256_CALB2024_dr0.1_FT_seed2024-CPMLP/

# FT_CALB_CPTransformer_checkpoints
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el6_dl0_df256_CALB42_dr0.1_FT_seed42-CPTransformer/
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el12_dl0_df256_CALB_dr0.1_FT_seed2021-CPTransformer/
# /data/hwx/PBT_transfer/CPTransformer_as128_al2_bs8_lr5e-05_dm128_el12_dl0_df256_CALB2024_dr0.1_FT_seed2024-CPTransformer/


batch_size=16
num_process=2
master_port=26945
eval_cycle_min=-1 # set eval_cycle_min as 1 and eval_cycle_max as 100 to evaluate all samples
eval_cycle_max=-1
eval_dataset=CALB42
model=CPMLP
finetune_method=FT # FT, AT
seed=42

CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_model_legacy.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model \
  --seed $seed \
  --finetune_method $finetune_method