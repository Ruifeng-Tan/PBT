args_path=/data/hwx/best_checkpoints/CPTransformer_sl1_lr5e-05_dm128_nh4_el12_dl0_df256_lradjconstant_datasetMIX_large_lossMSE_wd0.0_wlTrue_bs16_s2021-CPTransformer/
batch_size=128
num_process=2
master_port=24930
eval_cycle_min=1 # set eval_cycle_min or eval_cycle_max smaller than 0 to evaluate all testing samples
eval_cycle_max=100
eval_dataset=ISU_ILCC # ZN-coin, NAion, CALB
root_path=/data/trf/python_works/BatteryLife/dataset
LLM_path=/data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --LLM_path $LLM_path \
  --root_path $root_path
