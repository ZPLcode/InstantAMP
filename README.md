# InstantAMP

Train one policy per robot to walk, run and get up after a fall. A locomotion motion prior rewards gait style. A recovery motion prior provides the style signal on the floor. The learner fits a dynamics model from MuJoCo transitions and updates the policy through imagined rollouts and physical PPO samples.

| Robot | Training task (`--robot`) | Policy inputs | Actions |
| --- | --- | ---: | ---: |
| Unitree G1 | `g1` | 384 | 29 |
| Zenbot 23-DoF | `zenbot23_walk_run_recovery` | 234 | 23 |
| Zenbot 26-DoF | `zenbot26_walk_run_recovery` | 261 | 26 |

Both Zenbots share the training loop, command distribution, reward functions and evaluation criteria. Each uses three frames of Motion Train observations. Joint order, actuator calibration, link geometry and the dimensions derived from joint count follow the selected model. The 26-DoF model adds waist roll, waist pitch and head yaw. [Model definitions](instantamp/zenbot_models.py)

## Setup

The reference training environment uses Linux, Python 3.11, an NVIDIA GPU and a driver compatible with the pinned CUDA and Warp wheels. The G1 reference run used driver 580. Run the following from the checkout.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python -m pytest -q tests
```

## G1

Install Git LFS, Git and curl for motion preparation. The preparation script downloads the LeggedLab walk/run set and the six Unitree LAFAN1 fall-and-get-up takes, and converts them to 50 Hz NPZ files. Training uses primitive collision geometry. The GIF renderer also needs the Unitree surface meshes. [Motion sources and terms](THIRD_PARTY_NOTICES.md)

```bash
bash tools/download_data.sh
python -m instantamp.train --robot g1 --log_dir runs/g1_joint
python -m instantamp.evaluate runs/g1_joint --last 1
bash tools/download_meshes.sh
python -m tools.render runs/g1_joint
```

## Zenbot

The reference recipe uses the same 17 gait clips and 11 recovery clips for both robots. The 23-DoF preparation verifies converted clips against the checked-in SHA-256 manifest. The 26-DoF preparation retargets those clips using FRAX and records the source and output hashes. Both models use the same per-clip sampling weights. [Reference checksums](assets/zenbot_wm_reference.sha256) · [Recovery sampling weights](data/zenbot23_walk_run_recovery/recovery/sampling.json)

To reproduce these banks, supply a Motion Train checkout containing `zenbot_amp_frax_walk_sole8` and a licensed copy of the prepared LeggedLab `get_up_frax` bank. The latter must contain its manifest and motion pickles under `source/legged_lab/legged_lab/data/MotionData/zenbot_23dof_43kg/amp/get_up_frax`. Access to those source files is a prerequisite for the reference run. To generate the 26-DoF clips, set `FRAX_PYTHON` to an interpreter with `frax[ik]`, JAX, SciPy and MuJoCo. The retarget was tested with FRAX 0.2.0 and MuJoCo 4.0.1 in a separate environment. [Motion Train source revision](https://github.com/ZPLcode/h1_train/tree/4733fc9649dc3f2b99b2c3ba21fb2ec5e259e6ff)

```bash
export MOTION_TRAIN_CHECKOUT=/path/to/motion_train
export LEGGED_LAB_CHECKOUT=/path/to/legged_lab
# For 26-DoF data generation
export FRAX_PYTHON=/path/to/frax-env/bin/python

robot=zenbot23_walk_run_recovery  # or zenbot26_walk_run_recovery
run="runs/$robot"
bash tools/recipes/train_zenbot_joint.sh "$run" --robot "$robot"
```

The script prepares both motion banks for the selected robot, trains from scratch and evaluates the final checkpoint. By default both Zenbots use 2,048 environments, 40% floor resets, seed 42 and 20,000 updates.

| Shared Zenbot reward setting | Default |
| --- | ---: |
| Planar velocity tracking weight | 2.0 |
| Yaw-rate tracking weight | 1.5 |
| Shared recovery and gait AMP scale (`gait_ls_scale`) | 5.0 |
| Action-rate penalty weight (`action_rate_l2_weight`) | -0.05 |

A continuous weight blends recovery and gait using the current pelvis height and uprightness. Its height range is 0.44–0.58 m, and its uprightness range runs from a 60° tilt to 30°. The same weight controls the task terms, AMP styles and their training samples, and mirror consistency. The default `amp_lerp=0.5` keeps the task/AMP mixture fixed throughout recovery and locomotion. Both discriminators use the same least-squares score scale. Quiet-standing reward remains active at a zero command and fades as movement is requested. The lifting-progress reward still starts at 0.44 m with weight 0.5. [Training rewards](instantamp/sim.py) · [AMP mixing](instantamp/train.py)

The 23-DoF recipe also applies a mirror consistency loss to imagined frames with an active motion command, weighted by the same current posture signal. Its weight is `--imag_gait_mirror_weight 0.2` (26-DoF default `0`). The log records the unweighted error in `imag_gait_mirror_mse` and the mean posture weight of eligible imagined frames in `imag_gait_mirror_fraction`. Resuming keeps the saved weight, with zero for checkpoints that predate this loss. To enable it on an older run, pass `--imag-gait-mirror-weight 0.2` to the resume tool.

To override these defaults, pass `--wm_tracking_scale`, `--gait_ls_scale` or `--action_rate_l2_weight` to the Python entry point. Prepare the data first and use a fresh run directory for each model.

```bash
bash tools/recipes/prepare_zenbot_wm_data.sh --robot "$robot"
python -m instantamp.train --robot "$robot" --help
python -m instantamp.train --robot "$robot" --log_dir "runs/${robot}_experiment"
```

To initialize from a compatible Motion Train gait export, install `onnx` and `onnxruntime` and run `python -m tools.import_zenbot_gait INPUT.onnx OUTPUT.npz`. Pass `--init_policy OUTPUT.npz` when starting the joint recipe. The importer verifies the converted actions against the ONNX network.

### Evaluation and playback

The native Zenbot evaluator runs the same actor for every trial. It measures heading-aligned speed at 1.0 and 2.5 m/s, alternating foot support and flight periods during running. Each fallen start runs for ten seconds and is scored on the final second of supported standing. The native tools read the recovery bank from the run configuration. Use `--reference DIR` to override it. The training run also contains `config.json`, policy exports in `ckpt/`, and full learner checkpoints in `ckpt/full/`.

```bash
python -m tools.evaluate_zenbot_joint "$run" --out "$run/evaluation/final.json"
python -m tools.play_zenbot_control --checkpoint "$run"
```

The viewer starts in a neutral stance. Use W/S and A/D for linear velocity, Q/E for yaw, X to stop, F to push, R for the next fallen start and Space to pause. It can display the simulation geometry directly. Pass `--mesh-dir DIR` for a matching visual model. Both the viewer and scorer accept an individual policy NPZ. To resume training, pass a full checkpoint and a fresh output directory.

```bash
python -m tools.resume_zenbot_wm "$run/ckpt/full/0039.npz" "${run}_continued" --until 22000
```

These reward changes take effect during training; updating the code does not change an existing exported policy's actions during playback. When continuing a full checkpoint from the previous reward scheme, the learner initializes a fresh critic and critic optimizer: its two phase inputs now contain the current pose weight and its complement, and its target returns have changed. The actor, world model and motion discriminators are restored. Evaluate the continued policy before selecting it. Episode termination and the evaluator's supported-standing success criteria remain separate from the continuous training weight.

### Exported Zenbot23 checkpoint

The [74,000-update export](releases/zenbot23_continuous_74000/README.md) includes the policy, complete learner state, configuration, training metrics and native evaluation. Training was stopped at the user's request before the planned 100,000 updates. This checkpoint passed 22/22 fixed recovery trials and tracked a 1.0 m/s command at 0.97 m/s; it reached only 0.86 m/s for a 2.5 m/s command and did not pass the running test. See the export for playback and continuation commands.

## G1 reference results

<p align="center">
  <img src="figures/g1_se_8k_walk.gif" width="49%" alt="G1 walking and running with AMP style, following velocity commands">
  <img src="figures/g1_se_8k_getup.gif" width="49%" alt="G1 getting up from the floor">
</p>

![Episode length, velocity error and yaw-rate error over training time](figures/g1_se_8k_curves.png)

The GIFs and curves above come from a 60-minute run on one RTX 5090 with 8192 environments. The GIFs show its final checkpoint.
The curves are measured on 512 robots × 20 s per checkpoint, while the robots are up and commanded to move. Episodes are close to
full length (1000 steps) from about 20 minutes on. The velocity error falls to about 0.45 m/s at 30 minutes and 0.35 m/s at 60
minutes.

The table below is the reference run: one RTX 5090 with 4096 environments and the default flags. Compilation took 146 s before
training started, and training ran at 3.4 iterations per second.

| Training time | Episode length (of 1000 steps) | Episodes ending in a fall | Linear tracking (per s) | Get-ups (rose / down starts) |
|---|---|---|---|---|
| 5 min | 85 | 1.00 | 0.008 | 80 / 485 |
| 10 min | 152 | 1.00 | 0.021 | 100 / 117 |
| 15 min | 810 | 0.23 | 0.40 | 34 / 35 |
| 20 min | 949 | 0.07 | 0.70 | 40 / 40 |
| **30 min** | **935** | **0.10** | **0.73** | 27 / 33 |
| 60 min | 993 | 0.01 | 0.80 | 20 / 20 |

The robot learns to get up first (by about 10 minutes) and walks from about 15 minutes on. We rolled out the checkpoint from about
55 minutes for 20 s in 256 environments. It ran upright and symmetrically, with its arms swinging at its sides, at commands up to
2.7 m/s. Its median velocity error was 0.17 m/s, and it fell once.

These numbers come from one seed. Expect some spread between seeds and GPUs.

## Acknowledgements and citation

InstantAMP stands on the following works. If you use it, please cite them.

- **SuperMimic** (Liao et al., 2026): the method, a differentiable world model with analytic policy gradients through imagined
  rollouts. [Paper](https://arxiv.org/abs/2610.28480) · [Code](https://github.com/HybridRobotics/supermimic)
- **InstantTrack** by Huy-Quang Dao: G1 motion tracking in minutes on one GPU, the origin of much of the world-model training
  code (through BlinkMimic and XMimic). [Code](https://github.com/Huy-Quang-Dao/instantTrack)
- **XSoma** (Bohlinger and Peters, 2026): the JAX and MuJoCo Warp implementation that InstantAMP is built on.
  [Paper](https://nico-bohlinger.github.io/papers/2026_xmimic_xsoma.pdf) · [Code](https://github.com/nico-bohlinger/xsoma)
- **ODE Policy** (Jia et al., 2026): the ODE actor (`--ode 1`). [Paper](https://arxiv.org/abs/2611.01729) ·
  [Code](https://github.com/NTUMARS/ODE-POLICY)

Also thanks to:

- AMP (Peng et al., 2021, [arXiv:2104.02180](https://arxiv.org/abs/2104.02180));
- [mjlab](https://github.com/mujocolab/mjlab), for the G1 model and actuator constants;
- the code lineage of [XMimic](https://github.com/nico-bohlinger/xmimic) and [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed);
- [legged_lab](https://github.com/zitongbai/legged_lab) and [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), for the motion data;
- [MuJoCo](https://mujoco.org), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [NVIDIA Warp](https://github.com/NVIDIA/warp) and [JAX](https://github.com/jax-ml/jax).

InstantAMP is an independent project. It is not affiliated with or endorsed by the authors above.

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}

@misc{dao2026instanttrack,
  title        = {InstantTrack: Whole-Body Motion Tracking for the Unitree G1 in Minutes on One GPU},
  author       = {Dao, Huy-Quang},
  year         = {2026},
  howpublished = {\url{https://github.com/Huy-Quang-Dao/instantTrack}}
}

@misc{bohlinger2026xmimicxsoma,
  title  = {XMimic and XSoma: Learning Humanoid Motion Tracking and Locomotion in Under a Minute},
  author = {Bohlinger, Nico and Peters, Jan},
  year   = {2026},
  url    = {https://nico-bohlinger.github.io/papers/2026_xmimic_xsoma.pdf}
}

@article{jia2026odepolicy,
  title   = {ODE Policy: Robot Learning via Action Dynamics, Not Actions},
  author  = {Jia, Jindou and Li, Jingliang and Zuo, Kuangji and An, Tuo and Hu, Yuxuan and Zhao, Mengfei and Lyu, Bofan and Bai, Jiaqi and Zhu, Daoyuan and Dai, Hengyu and Zhang, Junpeng and Yang, Jianfei},
  journal = {arXiv preprint arXiv:2611.01729},
  year    = {2026}
}

@article{peng2021amp,
  title   = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  author  = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  journal = {ACM Transactions on Graphics},
  volume  = {40},
  number  = {4},
  year    = {2021}
}
```

## License

InstantAMP code is distributed under MIT. The robot descriptions, motion datasets and upstream components have their own terms. [Project license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Contributing](CONTRIBUTING.md)
