# Zenbot23 continuous-reward checkpoint — 74,000 updates

This export contains the latest complete checkpoint from the continuous recovery/gait reward run. It uses one actor, two AMP discriminators, a fixed 50/50 task/style mixture and persistent quiet-standing reward. The motion banks and lifting-progress reward were unchanged.

Training started from scratch with seed 42, 2,048 environments and 150 world-model warmup iterations on an RTX 5060 Ti. The requested target was 100,000 iterations. The user stopped the run after its last logged iteration of 74,550; the latest complete checkpoint was saved at **74,000**. Updates after that checkpoint are not included in this export.

## Native MuJoCo evaluation

These results evaluate this exported policy, with seed 42, no external assistance, no automatic resets and ten simulated seconds per trial. Recovery success requires standing posture throughout the last second and sufficient foot support. Gait speeds are measured over the final two seconds.

| Trial | Result |
| --- | --- |
| Recovery from fixed fallen starts | **22/22 passed** |
| Forward command 1.0 m/s | **0.971 m/s**, upright with alternating support; passed |
| Forward command 2.5 m/s | **0.861 m/s**, no qualifying flight periods; failed |

These fixed trials do not establish recovery from every possible pose or natural-looking gait. The running and high-speed tracking limitations remain. Full results are in [evaluation/native.json](evaluation/native.json).

## Use the export

Run the commands below from the repository root, using the environment described in the main README and requirements.txt. Prepare the original motion banks with the repository recipe; this export does not redistribute motion clips or visual meshes. [Source requirements and preparation](../../README.md#zenbot) · [Third-party notices](../../THIRD_PARTY_NOTICES.md)

```bash
python -m tools.play_zenbot_control --checkpoint releases/zenbot23_continuous_74000

JAX_PLATFORMS=cpu python -m tools.evaluate_zenbot_joint \
  releases/zenbot23_continuous_74000 \
  --out runs/zenbot23_continuous_74000_evaluation.json

python -m tools.resume_zenbot_wm \
  releases/zenbot23_continuous_74000/ckpt/full/74000.npz \
  runs/zenbot23_continuous_resumed --until 100000
```

Playback controls: W/S change forward speed, A/D change lateral speed, Q/E turn, X stops, R selects the next fallen start, F applies a push and Space pauses. An optional `--mesh-dir /path/to/zenbot/meshes` supplies visual geometry.

The full learner checkpoint includes the actor, world model, both discriminators, critic, normalizers, optimizer states and completed curriculum statistics. It uses reward version 1, so continuation with this version retains the critic. Replay buffers and live simulator states are initialized again when continuing, as in the normal trainer.

## Contents and provenance

- [ckpt/74000.npz](ckpt/74000.npz): policy export, copied byte-for-byte from the saved training checkpoint.
- [ckpt/full/74000.npz](ckpt/full/74000.npz): full learner, compressed losslessly; all 198 arrays were checked against the original.
- [config.json](config.json): training configuration with portable output/cache paths and `max_iterations=74000`, allowing directory-based playback to select this checkpoint.
- [provenance.json](provenance.json), [initialization.json](initialization.json) and [training_source.patch.gz](training_source.patch.gz): run settings, initial model hashes, software versions and the tracked changes applied to base commit `f6b3207294d843874b0a0ad4f0582dda5856498d` when training started.
- [training_metrics.jsonl.gz](training_metrics.jsonl.gz): every logged metric row through iteration 74,000; [training_metrics_at_export.json](training_metrics_at_export.json) is the final row.
- [motion_reference.sha256](motion_reference.sha256): the unchanged repository data manifest used for the run.
- [validation.json](validation.json): export and code verification results.

Verify package integrity from this directory with `sha256sum --check SHA256SUMS`.
