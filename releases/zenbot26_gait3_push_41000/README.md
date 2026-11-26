# Zenbot26 gait and recovery at 41,000 iterations

Training started with seed 42 and 2,048 environments, then resumed from iteration 4,000 with 4,096 environments. The training used three-frame gait AMP, ten-frame recovery AMP, Motion Train dynamics randomization, and external pushes. The run was stopped after saving iteration 41,000. Its cumulative simulator step count at that checkpoint was 2,556,297,216.

- [ONNX policy](zenbot26_instantamp_41000.onnx) exports the averaged actor at 50 Hz with 261 observation values, 26 actions in Motion Train joint order, and embedded observation normalization. [ONNX validation](validation.json)
- [Policy checkpoint](ckpt/41000.npz) contains both the live and averaged actors. The [full learner checkpoint](ckpt/full/41000.npz) also contains the world model, discriminators, critic, optimizers, and curriculum state.
- [Training metrics](training_metrics.jsonl.gz) contain one row every 50 iterations. `train_s` and `wall_s` are relative to each launch and restart after iteration 4,000. The [checkpoint metrics](training_metrics_at_checkpoint.json) and [4096-environment console log](train.log.gz) record the final training phase.
- [Configuration](config.json), [provenance](provenance.json), and [checksums](SHA256SUMS) document the export. Use the [motion checksums](motion_reference.sha256) and the [walk/run](walk_run_sampling.json) and [recovery](recovery_sampling.json) sampling files to match the references prepared separately for continuation.

With the motion banks prepared, resume to the original 100,000-iteration target from the repository root.

```bash
python -m tools.resume_zenbot_wm \
  releases/zenbot26_gait3_push_41000/ckpt/full/41000.npz \
  runs/zenbot26_from41000 --until 100000
```
