# CALB
# PBT-TL
# /data/LLMs/checkpoints/PBT_10_Llama_1_as16_al12_le50_bs32_lr5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_CALB42_guideFalse_LBFalse_lossMSE_wd0.0_wlFalse_dr0.0_gdff512_E5_GE5_K-1_SFalse_augFalse_dsr0.75_ffsTrue_MIX_large_FT_seed42-b2
# /data/LLMs/checkpoints/PBT_10_Llama_1_as16_al12_le50_bs32_lr5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_CALB_guideFalse_LBFalse_lossMSE_wd0.0_wlFalse_dr0.0_gdff512_E5_GE5_K-1_SFalse_augFalse_dsr0.75_ffsTrue_MIX_large_FT_seed2021-b2
# /data/LLMs/checkpoints/PBT_10_Llama_1_as16_al12_le50_bs8_lr5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_CALB2024_guideFalse_LBFalse_lossMSE_wd0.0_wlFalse_dr0.0_gdff512_E5_GE5_K-1_SFalse_augFalse_dsr0.75_ffsTrue_MIX_large_FT_seed2024-b2



args_path=/data/trf/finetune_AT_checkpoints/CPMLP_as128_al2_bs16_lr5e-05_dm128_el4_dl2_df256_CALB42_dr0.0_FT_seed42-CPT/
batch_size=128
num_process=1
master_port=24920
eval_cycle_min=1 # set eval_cycle_min or eval_cycle_max smaller than 0 to evaluate all testing samples
eval_cycle_max=100
eval_dataset=CALB42 # ZN-coin, ZN-coin42, ZN-coin2024, NAion, CALB. Note: For transfer learning variants, the evaluation dataset is automatically set as the target dataset
root_path=/data/trf/python_works/PBT_BatteryLife/dataset

LLM_path=/data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --LLM_path $LLM_path \
  --root_path $root_path
