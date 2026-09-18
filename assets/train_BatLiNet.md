# Training and evaluating BatLiNet

BatLiNet is integrated as a baseline that follows the original implementation: it is
trained on a *source* battery dataset and evaluated on a *target* dataset through
few-shot support sets. Both training and evaluation run through the same shared
scripts as the other models (`run_main.py` / `evaluate_model.py`); no separate
entry points are required.

## 1. Train BatLiNet

All commands below are run from the repository root, following the convention of
the other scripts (e.g., `sh scripts/PBT.sh`).

Edit the variables at the top of `scripts/BatLiNet.sh` and launch it:

```bash
sh scripts/BatLiNet.sh
```

The variables that matter for BatLiNet are:

| Variable | Meaning |
|---|---|
| `model_name` | keep as `BatLiNet` |
| `data` | keep as `Dataset_original` (the baseline data pipeline used by BatLiNet) |
| `dataset` | the **source domain** used for training (see below) |
| `target_dataset` | the **target dataset** whose validation/test splits are used for model selection and testing |
| `root_path` | path of the BatteryLife dataset directory |
| `checkpoints` | output directory for checkpoints |
| `in_channels`, `channels`, `input_height`, `input_width` | BatLiNet architecture hyperparameters |
| `max_cycle_index`, `diff_base`, `alpha` | BatLiNet feature-extraction hyperparameters |
| `train_support_size`, `test_support_size` | number of support samples per query during training / testing |

### Choosing the source domain

Two source domains are supported out of the box (both pick their validation and
test splits from `target_dataset`):

- **LFP as the source domain** — set

  ```bash
  dataset=LFP
  ```

  Training uses the LFP-cathode subset of the MIX_large training cells
  (`MIX_large_cathode_LFP_train_files` in `data_provider/data_split_recorder.py`).

- **MIX_large as the source domain** — set

  ```bash
  dataset=MIX
  ```

  Training uses the full MIX_large training cells (`MIX_large_train_files`).
  Note: inside the `Dataset_original` pipeline, the BatLiNet-compatible split is
  registered under the name `MIX` even though it loads the MIX_large training
  files; do not use `dataset=MIX_large` here, because that split takes its
  validation/test files from MIX itself instead of `target_dataset`.

`target_dataset` can be any of: `CALCE`, `HNEI`, `HUST`, `MATR`, `SNL`, `MICH`,
`MICH_EXP`, `RWTH`, `UL-PUR`, `Stanford`, `ISU_ILCC`, `XJTU`, `Tongji`,
`ZN-coin`, `ZN-coin42`, `ZN-coin2024`, `CALB`, `CALB42`, `CALB2024`, `NAion`,
`NAion42`, `NAion2024`.

### Model selection

Following the original BatLiNet implementation, checkpoints are selected by the
**validation MAPE** on the target-dataset validation split (see
`vali_baseline_with_BLN` in `utils/tools.py`), with early stopping. Each run
writes `model.safetensors`, `args.json`, and `label_scaler` into
`checkpoints/<setting>-<comment>/`.

## 2. Evaluate a saved BatLiNet checkpoint

Use the dedicated entry `scripts/evaluate_model_batlinet.sh`: set `args_path` to
the checkpoint directory produced by training and `eval_dataset` to the dataset
to evaluate on, then run from the repository root:

```bash
sh scripts/evaluate_model_batlinet.sh
```

This script simply drives the shared evaluator with `--model BatLiNet`, so the
general-purpose entry works identically (set `model=BatLiNet` in
`scripts/evaluate_model.sh`):

```bash
accelerate launch evaluate_model.py \
  --args_path <checkpoint_dir>/ \
  --batch_size 16 \
  --eval_cycle_min -1 \
  --eval_cycle_max -1 \
  --eval_dataset CALCE \
  --model BatLiNet \
  --root_path /path/to/BatteryLife/dataset
```

(`eval_cycle_min`/`eval_cycle_max` set to `-1` evaluates all testing samples;
set them to `1`/`100`, for example, to restrict evaluation to cells with 1–100
seen cycles.)

Evaluation reconstructs the support sets exactly as during training
(`get_support_set` with `training=False`, i.e., `test_support_size` supports per
query) and reports:

- cell-level MAPE and α-accuracy (15% / 10%),
- seen / unseen splits of the above,
- **condition-level MAPE** (macro-averaged over aging conditions, with
  seen/unseen condition breakdowns) — computed by the *same* metric routine
  (`condition_level_mape` in `evaluate_model.py`) and the *same* condition-ID
  mapping (`gate_data/name2agingConditionID.json`) as for every other model, so
  BatLiNet numbers are directly comparable with PBT and the CP baselines.

Detailed results are saved to `./results/` (`<model>_<dataset>_<seed>.json`,
plus per-cycle-number MAPE/accuracy JSONs); pass `--metrics_output` to also
write a concise metrics summary file.
