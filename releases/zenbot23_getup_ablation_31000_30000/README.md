# Zenbot23 recovery reward experiments

These are the latest complete checkpoints from two runs stopped on request at 10:07 on November 26, 2026 (Taipei). Both started from the same random weights with seed 42, 2,048 environments, Motion Train dynamics randomization, and 500 world model warmup iterations. Both used gait AMP scale 1.5, AMP mixing coefficient 0.7, and learned per-joint exploration standard deviations. The target was 100,000 total iterations per run.

| Run | Lift weight | Recovery AMP scale | Exported iteration | Last logged iteration |
| --- | ---: | ---: | ---: | ---: |
| [A](A_lift02_getupamp5) | 0.2 | 5 | 31,000 | 31,250 |
| [B](B_lift05_getupamp10) | 0.5 | 10 | 30,000 | 30,600 |

Each run includes the policy in `ckpt/`, the full learner in `ckpt/full/`, the training configuration, and compressed metrics and training logs through the last logged iteration. The learner checkpoint includes the world model, critics, motion discriminators, optimizers, and learned exploration. The full checkpoints were compressed losslessly and compared array by array with the training originals. The `max_iterations` value in the export config selects the saved checkpoint when opening the run directory. The full checkpoints contain no updates after their exported iteration.

## Native MuJoCo evaluation

Each checkpoint was tested on the same 22 fixed fallen starts on flat ground with zero commanded velocity, no assistance, and no resets. Recovery success requires upright posture throughout the final second and sufficient foot support. The timing metric measures the start of the first 0.2-second upright stretch within six seconds.

| Metric | A | B |
| --- | ---: | ---: |
| Recovery successes in 10 seconds | **22/22** | **4/22** |
| Starts that reached the upright timing criterion within six seconds | 22/22 | 2/22 |
| Median time to that criterion | 2.45 s | 2.09 s (2 starts) |
| Median peak upward pelvis speed before that criterion | 0.93 m/s | 0.55 m/s (2 starts) |
| Forward speed for a 1.0 m/s command | 0.97 m/s, passed | 1.08 m/s, passed |
| Forward speed for a 2.5 m/s command | 2.48 m/s, failed upright criterion | 2.44 m/s, failed upright criterion |

The matching kinematic reference clips reach the timing criterion at a median of 2.39 seconds with a median peak upward pelvis speed of 0.63 m/s. B's timing values describe only the two starts that reached the criterion before six seconds. These runs changed both lift weight and recovery AMP scale, and their checkpoints are at different iterations, so their comparison does not isolate either reward. [A evaluation](A_lift02_getupamp5/evaluation/native.json) · [B evaluation](B_lift05_getupamp10/evaluation/native.json)

## Playback and continuation

From the repository root, with the 23-DoF recovery references prepared, run:

```bash
release=releases/zenbot23_getup_ablation_31000_30000
python -m tools.play_zenbot_control --checkpoint "$release/A_lift02_getupamp5"
python -m tools.play_zenbot_control --checkpoint "$release/B_lift05_getupamp10"
```

To continue A from its complete checkpoint, prepare both motion banks and run:

```bash
python -m tools.resume_zenbot_wm "$release/A_lift02_getupamp5/ckpt/full/31000.npz" \
  runs/zenbot23_lift02_getupamp5_continued --until 100000
```

[Motion references and preparation](../../README.md#zenbot) · [Third-party notices](../../THIRD_PARTY_NOTICES.md)

[Provenance](provenance.json) records the source commit and training patch. [Motion reference checksums](motion_reference.sha256) list exactly the local files used for both runs. The training gait bank used the [archived sampling metadata](gait_sampling.json) with weighted clip sampling. The current data preparation recipe adds a marker for command-based sampling. Match the archived metadata in a separate gait bank if reproducing or continuing with the original reference sampling. The motion clips themselves are prepared separately.

Verify export integrity from this directory with `sha256sum --check SHA256SUMS`.
