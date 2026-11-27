# Zenbot26 gait and recovery at 55,000 iterations

This 26-DoF policy trained from random initialization with seed 42, 2,048 environments and 500 world-model warmup iterations. Gait AMP used a multiplier of 3 and get-up AMP used 10, with an AMP mixture of 0.7. The training enabled external pushes, a recovery control scale of 1 and a default standing-pose weight of −2. [Training and export code](https://github.com/ZPLcode/InstantAMP/tree/1508cd342753531500394a267533cd0ac54e8d1a)

The training service stopped at the user's request on November 27, 2026. The original target was 100,000 iterations. The last complete checkpoint is iteration 55,000 at 1,802,371,072 simulator steps. The training log extends to iteration 55,200. Checkpoints were saved every 1,000 iterations. [Training configuration](training_config.json) · [Final checkpoint metrics](training_metrics_at_checkpoint.json) · [Training metrics through the stop](training_metrics.jsonl.gz) · [Provenance](provenance.json)

## Native MuJoCo results

The averaged actor ran at 50 Hz on flat ground with seed 42. Walking and running speeds are heading-aligned averages over the final two seconds of each ten-second trial. Recovery success requires a sustained supported stance at the end of the trial. The push trials used a 0.4-second torso force of 300, 450 or 600 N in each of four directions. [Per-trial results and effort measurements](evaluation.json)

| Trial | Result |
| --- | --- |
| Walk at 1.0 m/s | 1.15 m/s, upright with alternating steps. Passed. |
| Run at 2.5 m/s | 0.02 m/s, lost upright posture. Failed. |
| Fixed fallen starts | 20 of 22 recovered, 19 on the first attempt. |
| Push-induced falls | 12 of 12 recovered, 9 on the first attempt. |

## Policy and checkpoints

The ONNX actor accepts a float32 `obs[1,261]` with three frames grouped by observation term and includes observation normalization. It outputs float32 `actions[1,26]` in Motion Train joint order. Joint names and default joint angles are embedded in the model metadata. ONNX Runtime outputs were compared with the averaged native actor. [ONNX policy](zenbot26_instantamp_55000.onnx) · [Output parity](validation.json)

The policy checkpoint contains the live and averaged actors. The full checkpoint also restores the world model, AMP discriminators, critic, optimizer and curriculum state. The release configuration selects iteration 55,000 for playback. The training configuration records the original target and uses a repository-relative cache path. [Policy checkpoint](ckpt/55000.npz) · [Full learner checkpoint](ckpt/full/55000.npz) · [Release configuration](config.json) · [Checksums](SHA256SUMS)

From the repository root, open the viewer, reproduce the evaluation, or resume into a new run directory.

```bash
release=releases/zenbot26_scratch_55000
python -m tools.play_zenbot_control --checkpoint "$release"
python -m tools.evaluate_zenbot_joint "$release" --push-recovery \
    --out runs/zenbot26_55000_evaluation.json
python -m tools.resume_zenbot_wm "$release/ckpt/full/55000.npz" \
    runs/zenbot26_from55000 --num-envs 2048 --until 100000
```

To reproduce the original training from iteration zero with the checked-in motion banks, use a fresh output directory.

```bash
python -m instantamp.train --robot zenbot26_walk_run_recovery \
    --num_envs 2048 --max_iterations 100000 --minutes 0 --seed 42 --wm_only_iters 500 \
    --wm_gait_frames 3 --gait_ls_scale 3 --getup_amp_scale 10 --amp_lerp 0.7 --external_pushes 1 \
    --wm_recovery_control_scale 1 --wm_stand_pose_weight -2 --wm_lift_weight 0.2 \
    --imag_gait_mirror_weight 0.2 --ckpt_iterations 1000 --full_ckpt_keep 2 \
    --log_dir runs/zenbot26_scratch_reproduction
```

To reproduce the ONNX export, install `onnx` and `onnxruntime` in the project environment and run the exporter on the release policy.

```bash
python -m tools.export_zenbot_onnx "$release/ckpt/55000.npz" \
    runs/zenbot26_55000.onnx --validation runs/zenbot26_55000_onnx_validation.json
```
