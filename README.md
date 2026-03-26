# AutoRewardDrive

## Clean Re-Train

Run a fresh `SAC_AUTO_V2` experiment from scratch instead of resuming the interrupted
`shield_smooth_v1` run.

Recommended command:

```bash
conda run -n vlm-rl python train.py \
  --config 4 \
  --total_timesteps 500000 \
  --device cuda:0 \
  --host localhost \
  --port 2000 \
  --fps 15 \
  --no_render \
  --start_carla \
  --num_checkpoints 50 \
  --latest_bundle_freq 50000 \
  --run_name shield_smooth_v2_clean
```

Equivalent helper script:

```bash
bash scripts/train_shield_smooth_v2.sh
```

Override defaults with environment variables when needed:

```bash
RUN_NAME=my_run DEVICE=cuda:1 PORT=2100 bash scripts/train_shield_smooth_v2.sh
```

## Late Checkpoint Eval

After training, compare raw-policy and shielded-policy performance on the same
set of late checkpoints.

Recommended raw eval:

```bash
conda run -n vlm-rl python run_eval.py \
  --model_dir tensorboard/AutoRewardedSACV2_shield_smooth_v2_clean_id4 \
  --models model_150000_steps.zip model_200000_steps.zip model_300000_steps.zip model_500000_steps.zip \
  --config 4 \
  --device cuda:0 \
  --town Town02 \
  --density regular \
  --port 2020 \
  --seed 101 \
  --episodes 10 \
  --inference_mode step \
  --eval_tag step_raw
```

Recommended shielded eval:

```bash
conda run -n vlm-rl python run_eval.py \
  --model_dir tensorboard/AutoRewardedSACV2_shield_smooth_v2_clean_id4 \
  --models model_150000_steps.zip model_200000_steps.zip model_300000_steps.zip model_500000_steps.zip \
  --config 4 \
  --device cuda:0 \
  --town Town02 \
  --density regular \
  --port 2020 \
  --seed 101 \
  --episodes 10 \
  --use_shield \
  --inference_mode step \
  --eval_tag step_shielded
```

Equivalent helper script:

```bash
bash scripts/eval_shield_smooth_v2.sh tensorboard/AutoRewardedSACV2_shield_smooth_v2_clean_id4
```

## Metrics To Watch

Training:

- `rolling/success_rate_50`
- `rolling/route_completion_50`
- `rolling/collision_rate_50`
- `smooth/actor_smooth_loss`
- `smooth/mean_steer_delta`
- `shield/train_intervention_rate`
- `shield/bc_loss`
- `autoreward/reward_corr_ema`
- `autoreward/reward_ready`

Evaluation:

- `success`
- `routes_completed`
- `center_dev_mean`
- `CPS`
- `CPM`
- `shield_intervention_rate`

## Notes

- Keep `action_smoothing = 0.0`
- Do not use chunked execution as the primary selection criterion
- Judge a checkpoint using both raw and shielded results
- `model_final.zip` from interrupted runs may be stale; prefer milestone checkpoints
