# InstantAMP: Train AMP in 30 minutes on 1 RTX 5090

InstantAMP trains a Unitree G1 velocity-tracking policy with an **adversarial motion prior (AMP)** in about 30 minutes on a single
RTX 5090. The policy walks, runs, side-steps and turns like the human clips it learns from, and it gets back up after a fall.

The policy is not trained with model-free RL alone. Like [SuperMimic](https://github.com/HybridRobotics/supermimic), InstantAMP fits a
**differentiable world model** of the robot from simulator transitions. The policy is then optimized with **analytic gradients through
short imagined rollouts** (backpropagation through time). The style rewards of the two motion priors are differentiable in the
imagined states too, so the policy follows their gradient through the world model like any other reward term.

The whole training loop is compiled with JAX, including the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) simulator,
and runs on one GPU. The code builds on [XSoma](https://github.com/nico-bohlinger/xsoma).

## Contents

- [What is trained](#what-is-trained)
- [Results](#results)
- [Setup](#setup)
- [Motion data](#motion-data)
- [Training](#training)
- [Flags](#flags)
- [Evaluation and rendering](#evaluation-and-rendering)
- [Repository layout](#repository-layout)
- [Acknowledgements and citation](#acknowledgements-and-citation)
- [License](#license)

## What is trained

| Part | What it is |
|---|---|
| Robot | Unitree G1, 29 joints. mjlab's MJCF with full collision (self-collisions included) and the Unitree actuator model (torque-speed curve and friction), 200 Hz physics, 50 Hz policy. |
| Task | Follow velocity commands on flat ground: vx ∈ [−1.5, 3.0] m/s, vy ∈ [−1, 1] m/s, wz ∈ [−π/2, π/2] rad/s. 25 % of the envs use heading control and 5 % stand still. Commands are redrawn every 3 to 8 s. |
| Observation | 4 frames of history (384 values): pelvis angular velocity, gravity, command, joint positions and velocities, last action. Noise is added, and the policy normalizes the observation with running statistics. |
| Rewards | Velocity tracking on the pelvis, regularizers (joint acceleration, joint limits, action rate, foot slip, self-collision, wrist pose), a termination penalty, and a rise penalty while the robot is down. |
| Falls and get-up | An upright robot is terminated as soon as its pelvis, hips, shoulders, torso or hands touch the floor. 20 % of the resets start on the floor in a frame of a LAFAN1 get-up clip, and the robot has 5 s to stand up. |
| Robustness | Pushes every 5 to 10 s, shoves at reset, randomized foot friction, encoder bias and torso CoM. |
| Motion priors | Two WGAN-GP discriminators over 10-frame windows: one for walking and running (used while the robot is up) and one for getting up (used while it is down). |
| Learner | World model (MLP 3×512, 8-step open-loop loss), BPTT over a 48-step horizon (about 1 s), a critic that bootstraps the return at the horizon, an ODE actor, and a hybrid PPO term on 10 % of the real collector steps. |

The reward that the policy maximizes in imagination is

```
r = (1 − amp_lerp) · r_task + amp_lerp · dt · (ramp · gate_loco · style_loco + gate_getup · style_getup)
```

- `gate_loco = moving · (0.25 + track_lin)` while the robot is up, so tracking well earns more style reward. It is 0 for a standing command.
- `gate_getup = 1` while the robot has not been up yet in its episode.
- Each style is `(1 + tanh((d − mean robot score) / 5)) / 2`. Centering it on the robot's own mean score keeps the reward slope constant, however far the robot still is from the demos.

## Results

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

## Setup

You need Linux with an NVIDIA GPU, Python 3.11, and an NVIDIA driver with CUDA 12.8 support or newer. Warp's PyPI wheel needs
driver 580 or newer; the reference run used 580.

```bash
git clone https://github.com/RoboticsResearchHubVN/InstantAMP.git
cd InstantAMP
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

With an older driver, install Warp's CUDA 12 build (`warp_lang-1.19.0+cu12`) from
[its releases](https://github.com/NVIDIA/warp/releases) instead of the PyPI wheel.

Check the install on the CPU (no GPU or data needed):

```bash
JAX_PLATFORMS=cpu python -m pytest tests
```

## Motion data

The clips are not part of this repository; their own licenses apply. One script downloads them and converts them to 50 Hz `.npz`
files in the G1's joint order. It needs `git`, `git-lfs` and `curl`.

```bash
bash tools/download_data.sh
```

| Folder | Clips | Used for | Source |
|---|---|---|---|
| `data/amp/walk_run` | 30 clips, 156 s: walking, turning, side-stepping, running, running backwards, transitions | demos of the locomotion prior, upright reset frames | [legged_lab](https://github.com/zitongbai/legged_lab) (GMR retargets of AMASS / ACCAD) |
| `data/recovery/lafan1` | 6 `fallAndGetUp` takes, 935 s (24 056 ten-frame windows in which the robot is down) | demos of the get-up prior, floor reset frames | [Unitree's LAFAN1 retarget](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (LAFAN1 by Ubisoft La Forge, CC BY-NC-ND 4.0) |

To use your own clips, convert them with [`tools/convert_motions.py`](tools/convert_motions.py) and pass their folders with
`--amp_dir` / `--walk_dir` (locomotion) or `--amp_getup_dir` / `--floor_dir` (get-up). The script reads legged_lab pickles
(`amp`) or Unitree-style CSV files (`csv`: root position, root quaternion x y z w, then the 29 joints, 30 fps by default). An
`.npz` clip holds `fps` (50), `root_pos`, `root_quat` (x, y, z, w), `root_lin_vel`, `root_ang_vel` (world frame), `dof_pos` and
`dof_vel` (29 joints in the order of `instantamp/env.py`'s `DOF`).

## G1 recovery style

The G1 recovery experiment uses the original 29-joint actor and native 50 Hz motions.
The preparation tool extracts 61 forward rises from five takes. It reserves the entire
`fallAndGetUp1_subject1` take for evaluation of the fine-tuning. The original G1
checkpoint was trained on all six untrimmed takes.

```bash
python tools/prepare_g1_rises.py
python -m instantamp.train --g1_style 1 --num_envs 4096 \
  --init_g1_full runs/g1_4096_20261120_01/ckpt/full/0011.npz \
  --buffer_steps 128 --amp_batch 1024 --wm_batch 768 --pi_batch 640 \
  --wm_only_iters 80 --pi_ema 0.99 --max_iterations 850 --minutes 0 \
  --ckpt_minutes 1000 --log_dir runs/g1_style_replica
python tools/export_g1_arm_blend.py runs/g1_4096_20261120_01/ckpt/0011.npz \
  runs/g1_style_replica/ckpt/0000.npz --out runs/g1_arm_recovery
python tools/evaluate_g1_style.py runs/g1_4096_20261120_01/ckpt/0011.npz \
  runs/g1_arm_recovery/ckpt/best.npz --out runs/g1_arm_recovery/comparison.json \
  --gif runs/g1_arm_recovery/comparison.gif
```

Fine-tuning starts from the averaged baseline actor and copies the world model,
observation statistics, critic, and walking discriminator. A new least-squares recovery
discriminator observes ten consecutive frames including pelvis height and vertical
velocity. It scores the first four seconds of a floor-start episode. Half the resets
come from the opening 0.4 seconds of a recovery. The commands above save the final
actor at iteration 850. In the original run, this actor is checkpoint 0001.

The exported policy blends up to 20% of the fine-tuned actor's shoulder, elbow and
wrist action increments into the baseline. Projected gravity and knee flexion set
the blend at each step. The baseline controls the waist and legs. Both actors use
their own observation statistics and share one action integrator. Load the export
through `instantamp.inference.load` and call `policy_output` before the usual ODE
integration. The native renderer and evaluator also accept the exported run directory.

On the eleven fixed starts from the reserved take, both policies stood up within
five seconds and stayed upright through seconds five to six. Comparing corresponding
fractions of the rise, median wrist error fell from 37.3 to 36.4 cm, ankle error
from 56.7 to 55.6 cm, and joint error from 0.704 to 0.683 rad. The reference took
2.34 s to rise at the median. The original and exported policies took 0.82 and
0.80 s. The comparison uses nominal physics and zero assistance.

### Following a specific G1 get-up

For a specified sequence of poses at 50 Hz, run the reference controller from
the repository root. Use the native G1 checkpoint to take over balance after
the last reference frame. Choose a new output directory for each run.

```bash
.venv/bin/python tools/plan_g1_reference.py --fixed --out runs/g1_reference_check --gif
# A single native G1 get-up, starting on the floor and ending in stance
.venv/bin/python tools/plan_g1_reference.py --clip path/to/getup.npz \
  --balance runs/g1_4096_20261120_01/ckpt/0011.npz --out runs/g1_one_reference
```

At each replan, 200 batched MuJoCo Warp predictions start from the measured state.
They evaluate 0.48 s of joint, pelvis and landmark tracking. A separate native
MuJoCo simulation executes three 20 ms control steps, then supplies its new state.
The robot uses its original motor torque and speed limits. Hand and knee contacts
with the floor can carry tangential force. The opt-in contact setting is scoped
to the reference task. The simulation initializes the floating base once.

The output NPZ contains the reference and simulated poses, velocities, 50 Hz
joint targets, and the actual actuator inputs for each 5 ms substep. The JSON
report compares trajectories at equal times from the fallen start through 0.5 s
after the reference first reaches the standing threshold. For the eleven fixed
starts, all eleven stood up unassisted and stayed upright throughout the final
second. Median across-trial RMS errors were 2.3 cm in pelvis height, 5.7° in
pelvis orientation, 6.2 cm at the wrists and 5.5 cm at the ankles in the world
frame. The largest rise-time difference was a single 20 ms control step.

For repeated execution of a planned clip, export the nominal joint targets and
a time-varying state-feedback gain for each 20 ms control step. The exported
controller uses NumPy at execution time and takes the measured floating-base
and joint state in the plan's coordinate system. A new clip needs its own plan.
The exported eleven-clip bank stood up and held stance on all eleven nominal
starts. It also held stance on all 22 validation starts with paired initial
height and horizontal speed offsets of ±5 mm and ±0.08 m/s. Across the nominal
trials its median wrist and ankle errors were 6.3 and 5.5 cm.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/export_g1_tracking_policy.py \
  runs/g1_reference_check/trajectories_000.npz --out runs/g1_getup_feedback
```

```python
from instantamp.timed_feedback import TimedFeedbackPolicy

policy = TimedFeedbackPolicy.load("runs/g1_getup_feedback/policy.npz")
case = policy.names.index("fallAndGetUp1_subject1_458_565")
# qpos and qvel come from the robot at this control step.
joint_target = policy.target(case, tick, qpos, qvel)
```

The repository also supports the LAFAN1 anchor-conditioned SuperMimic G1 neural
policy if that checkpoint is available locally. Its inputs are ten reference
frames, the robot's sensor history and the reference torso pose relative to the
current torso. The export step uses the source checkout's PyTorch environment.

```bash
python tools/track_g1_reference.py export /path/to/ckpt_latest.pt g1_reference_actor.npz
.venv/bin/python tools/track_g1_reference.py run g1_reference_actor.npz \
  data/recovery/lafan1/fallAndGetUp1_subject1.npz --start 763 --end 942 \
  --seconds 6 --out runs/g1_actor_reference.npz
```

## Zenbot 23-DoF get-up

The `zenbot23_getup` task uses the 43.0678308 kg articulation from
`motion_train/robolab/data/robots/roboparty/zenbot`. Recovery demonstrations and task settings come from the local
`legged_lab` configuration `tasks/locomotion/amp/config/zenbot/zenbot_get_up_env_cfg.py` and its G1 get-up base class.

```bash
python tools/prepare_zenbot.py /path/to/motion_train --legged-lab /path/to/legged_lab --meshes
python -m instantamp.train --robot zenbot23_getup --num_envs 4096
python -m instantamp.evaluate runs/zenbot23_getup --last 1
python tools/render.py runs/zenbot23_getup --out runs/zenbot23_getup/preview
```

Preparation verifies the MJCF and clip checksums against the `get_up_frax` manifest, maps named joints into MuJoCo hinge order,
and resamples the 11 selected 30 Hz recoveries to 50 Hz. Meshes are optional for training.
Resets sample the clips equally, center the pelvis in XY and start at the first frame with a 2 cm lift. Each episode lasts 10 s.
The 23 delayed PD motors accept joint offsets scaled by 0.25 rad, with an independently sampled 3–7 physics-step delay per actuator group.
The training colliders use the source URDF capsule dimensions and sole boxes. The actor sees five 96-value frames including the ankle,
knee and hand positions in the pelvis frame. The discriminator sees ten 73-value frames in the same landmark order.

The nine get-up reward terms follow the reference weights. Standing terms activate above 0.58 m and target a pelvis height of 0.68 m.
World-up torso assistance starts at 258.35 N. It requires more than 1 N of upward ground support on an ankle, knee or hand,
fades linearly from 0.48 to 0.68 m and drops by 12.92 N for the resetting environments when their mean episode height reward
exceeds 60% of its maximum. The reference-style discriminator uses LSGAN, a 1024/512 leaky-ReLU network, a style scale of 50 and
a task/style mix of 0.5. One discriminator covers the entire recovery including the standing phase.

The Zenbot recipe also tracks the selected reset clip at 50 Hz until its last frame.
The reference reward compares pelvis height and tilt, the 23 joint angles and the six landmark positions
in pelvis coordinates. The default weight is 5 per second before task/style mixing. Use
`--reference_weight 0` to reproduce the previous reward. Both real PPO samples and world-model rollouts
use the episode's clip and elapsed time. The actor uses its existing proprioceptive history at deployment.

The policy optimizer remains InstantAMP's imagined rollouts, world-model gradients and real PPO minibatches.
The real torque penalty reads the last physics substep. Imagination estimates it from delayed targets and predicted joint state.
The default limit is 100,000 InstantAMP iterations, with one real step per environment per iteration.
The reference PPO runner collects 24 real steps per environment per iteration. Set `--minutes` or `--max_iterations` for a shorter run.
Evaluation measures unassisted recovery over the final second on fixed fallen starts.
It also reports pelvis height MAE (cm), joint RMSE (rad) and landmark RMSE (cm) over the valid
reference frames, aligned to each evaluation start's 0 or 0.5 s offset.

## Training

```bash
python -m instantamp.train --log_dir runs/instantamp            # 30 minutes, the reference recipe
```

Every option of `Config` in [`instantamp/train.py`](instantamp/train.py) is a command-line flag of the same name
(`--name value`). Some common variations:

```bash
python -m instantamp.train --minutes 60 --log_dir runs/long                    # train longer
python -m instantamp.train --num_envs 2048 --log_dir runs/small                # less GPU memory
python -m instantamp.train --amp_lerp 0.3 --amp_style_scale 10 --log_dir runs/weak_style   # weaker style
python -m instantamp.train --amp_getup 0 --reset_floor 0 --log_dir runs/no_getup          # locomotion only
python -m instantamp.train --amp_getup_full 1 --imag_down 0.25 --log_dir runs/getup      # stronger get-up learning
```

A run writes the following to `--log_dir`:

- `config.json`: every option of the run.
- `log.jsonl`: one line every 50 iterations.
- `ckpt/NNNN.npz` and `ckpt/NNNN.json`: the averaged and live policies with the observation normalizer, every 2 minutes and at the end.
- `ckpt/full/NNNN.npz`: the whole training state.

The console line reads:

```
it 4300 | 20.13 min | 17.7M | len 949 fell 0.07 lin 0.700 ang 0.779 r 0.0362 | up 0.99 get-ups 40/40 | wm ... | G ... | 3.4 it/s
     | h_err 0.3/1.2 cm term env ... wm ... | amp d demo/robot r use ramp | getup d ... | V end ... | ppo clip ...
```

| Field | Meaning |
|---|---|
| `len` | mean length of the finished episodes (1000 = a full 20 s) |
| `fell` | share of the finished episodes that ended by termination |
| `lin`, `ang` | tracking terms summed over an episode, divided by 20 s (1.0 is perfect tracking) |
| `get-ups x/y` | of the episodes that started on the floor, how many rose |
| `wm`, `e8`, `h_err` | world-model loss and its open-loop pelvis-height error after 8 / 32 steps |
| `G`, `r_im`, `surv` | imagined return, imagined reward per step, share of imagined rollouts that survive the horizon |
| `amp d demo/robot` | mean discriminator scores of the demos and of the robot; `r` is the AMP reward per step |

The JAX compilation cache (`--jax_cache`, `~/.cache/instantamp/jax`) makes later starts much faster.

## Flags

All flags with their defaults (the reference recipe). Booleans are `0` / `1`.

**Environment**

| Flag | Default | Meaning |
|---|---|---|
| `--num_envs` | 4096 | parallel simulated robots |
| `--nconmax`, `--njmax` | 64, 640 | contacts and constraint rows per world (MuJoCo Warp pools) |
| `--walk_dir` | `data/amp/walk_run` | clips whose frames are the upright reset states |
| `--floor_dir` | `data/recovery/lafan1` | get-up clips whose down frames are the floor reset states |
| `--reset_floor` | 0.2 | share of resets that start on the floor |
| `--prob_shove` | 0.2 | share of upright resets that get a random kick of root and joint velocities |
| `--push` | 1 | overwrite the root's horizontal velocity with U[−1, 1] m/s every 5 to 10 s |
| `--random_start` | 1 | random initial episode counters, so that episodes do not all end together |
| `--term_tau` | 0.02 | width (m) of the sigmoid that gives imagined terminations their gradient |

**Motion priors (AMP)**

| Flag | Default | Meaning |
|---|---|---|
| `--amp_w` | 2.0 | `> 0` turns AMP on; with `--amp_lerp 0` it is the style weight |
| `--amp_lerp` | 0.5 | share of the style in the reward: `(1 − lerp)·task + lerp·dt·style`; `0`: `task + amp_w·dt·style` |
| `--amp_style_scale` | 5.0 | scale of the style's tanh around the robot's mean score (smaller = stronger style); `0`: midpoint / half-gap of the two mean scores |
| `--amp_gate` | `track_shift` | gate of the locomotion style: `track_shift` (moving × (lo + track_lin)), `moving`, `track` (moving × track_lin) or `none` |
| `--amp_track_lo` | 0.25 | the floor `lo` of the `track_shift` gate |
| `--amp_dir` | `data/amp/walk_run` | demos of the locomotion prior |
| `--amp_getup` | 1 | second discriminator for the get-up style |
| `--amp_getup_dir` | `data/recovery/lafan1` | demos of the get-up prior (windows in which the robot is down) |
| `--amp_getup_full` | 0 | `1`: the get-up style is at full scale from the first iteration, instead of sharing the ramp |
| `--amp_frames` | 10 | frames per discriminator window (0.2 s) |
| `--amp_wrists` | 1 | the wrist joints are part of the discriminator's frame |
| `--amp_disc_steps` | 2 | discriminator steps per iteration (each prior) |
| `--amp_batch` | 4096 | windows per discriminator step |
| `--amp_recent` | 100 | robot windows come from the newest this many buffer steps |
| `--amp_lr` | 1e-4 | discriminator learning rate |
| `--amp_delay`, `--amp_ramp` | 500, 4000 | the style reward starts after `delay` iterations and ramps up over `ramp` iterations |

**World model**

| Flag | Default | Meaning |
|---|---|---|
| `--wm_hidden` | `512,512,512` | hidden widths |
| `--wm_window` | 8 | open-loop steps of the loss |
| `--wm_batches`, `--wm_batch` | 3, 2048 | gradient steps per iteration and windows per step |
| `--wm_lr` | 4e-4 | learning rate |
| `--wm_muon` | 1 | Muon on the hidden weight matrices (Adam elsewhere) |
| `--wm_gate` | 2.0 | skip the policy update while the loss is above this times its running mean (0: off) |
| `--wm_eval` | 1 | log open-loop errors and termination rates |

**Policy**

| Flag | Default | Meaning |
|---|---|---|
| `--horizon` | 48 | imagined steps per rollout (about 1 s) |
| `--pi_batch` | 2048 | imagined rollouts per update |
| `--pi_updates` | 2 | policy updates per iteration |
| `--pi_lr` | 3e-4 | learning rate |
| `--pi_clip` | 1.0 | gradient-norm clip |
| `--pi_gn_gate` | 10.0 | skip an update whose gradient norm is above this times its running mean (0: off) |
| `--pi_ema` | 0.995 | decay of the averaged policy that checkpoints hold |
| `--gamma` | 0.99 | discount |
| `--ode` | 1 | ODE actor ([ODE Policy](https://arxiv.org/abs/2611.01729)): the network outputs the change of the action, integrated from the previous action |
| `--ode_dmax`, `--ode_amax` | 3.24, 20.0 | bounds of the action change per step and of the action (0: none) |
| `--action_noise` | 0.3 | exploration noise (std) of the noisy collector envs |
| `--det_share` | 0.5 | share of collector envs without noise (their windows train the discriminators) |
| `--cmd_fresh` | 0.5 | share of imagined rollouts that start with a freshly drawn command |
| `--imag_down` | 0.0 | share of imagined rollouts that start in a down state (0: uniform) |
| `--obs_clip` | 10.0 | clip of the normalized observation |

**Critic**

| Flag | Default | Meaning |
|---|---|---|
| `--critic` | 1 | value function that bootstraps the imagined return at the horizon (0: pure BPTT) |
| `--critic_hidden` | `512,512,256` | hidden widths |
| `--critic_lr`, `--critic_epochs`, `--critic_mb` | 3e-4, 5, 4 | TD(λ) fit on each batch of imagined rollouts |
| `--lam` | 0.95 | λ of the TD(λ) targets and of PPO's GAE |

**Hybrid PPO**

| Flag | Default | Meaning |
|---|---|---|
| `--ppo_frac` | 0.1 | share of the policy gradient's samples taken from real collector steps (0: off) |
| `--ppo_len` | 8 | newest steps of the PPO envs per batch |
| `--ppo_clip` | 0.2 | PPO ratio clip |
| `--ppo_w` | 1.0 | the PPO gradient is scaled to `ppo_w` times the norm of the BPTT gradient (< 0: p / (1 − p)) |

**Schedule and runtime**

| Flag | Default | Meaning |
|---|---|---|
| `--minutes` | 30.0 | training time, without compilation |
| `--env_steps` | 1 | simulator steps of all envs per iteration |
| `--buffer_steps` | 200 | replay buffer length, in steps of all envs |
| `--warmup_steps` | 32 | steps collected before learning |
| `--wm_only_iters` | 200 | first iterations fit only the world model |
| `--imag_unroll` | 4 | unroll of the imagination scan |
| `--imag_remat` | `none` | `dots` or `full`: rematerialize imagined steps (less memory, slower) |
| `--mm_bf16` | `imag` | bf16 matmul outputs: `imag`, `all` or `none` |
| `--log_every` | 50 | iterations per log line |
| `--ckpt_minutes` | 2.0 | minutes between checkpoints |
| `--seed` | 0 | random seed |
| `--log_dir` | `runs/instantamp` | output folder |
| `--jax_cache` | `~/.cache/instantamp/jax` | JAX compilation cache (empty: off) |
| `--xla_flags` | `--xla_gpu_enable_triton_gemm=false` | extra XLA flags, comma-separated |

JAX takes 30 % of the GPU's memory by default, and MuJoCo Warp allocates the rest at run time. Set
`XLA_PYTHON_CLIENT_MEM_FRACTION` to change JAX's share.

## Evaluation and rendering

```bash
python -m instantamp.evaluate runs/instantamp --last 3       # 1024 robots, 20 s, per checkpoint
```

For each checkpoint the evaluation prints and writes to `eval.jsonl`:

- the reward terms per second;
- the mean velocity error while up;
- the share of upright starts that fell;
- the share of floor starts that rose.

To render a walking GIF and a get-up GIF of a checkpoint (needs OpenGL; the 19 MB of visual meshes are downloaded once):

```bash
bash tools/download_meshes.sh
python tools/render.py runs/instantamp --out figures/instantamp     # figures/instantamp_walk.gif, _getup.gif
```

To check that imagination's reward terms match the simulator's on real transitions (needs a GPU):

```bash
python tools/check_heads.py
```

## Repository layout

```
instantamp/
  env.py         robot, actuators, commands, observations, rewards, terminations: pure JAX, shared by simulator and imagination
  sim.py         MuJoCo Warp simulator: scene, physics, contacts, resets, pushes, forward kinematics with velocities
  train.py       the learner: collection, motion priors, world model, BPTT (+ critic, PPO), checkpoints; Config = all flags
  evaluate.py    evaluation of checkpoints
  amp.py         AMP discriminator (WGAN-GP), frame features, style reward
  getup_amp.py   demo windows of the get-up prior
  nets.py        world model and policy networks, ODE actor, running normalizers
  critic.py      value function and its TD(lambda) fit
  buffer.py      device replay buffer
  resets.py      reset banks from motion clips
  optim.py       MuonAdam
  kin.py, rot.py forward kinematics and rotations
assets/g1_mjlab.xml   the G1 MJCF (visual meshes: tools/download_meshes.sh)
tools/         download_data.sh, convert_motions.py, download_meshes.sh, render.py, check_heads.py
tests/         CPU tests: actuator and kinematics against MuJoCo, shapes
```

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

InstantAMP is released under the [MIT License](LICENSE). It derives from XSoma (MIT), whose copyright notice is kept. The bundled G1
MJCF comes from mjlab (Apache-2.0) and derives from Unitree's BSD-3-Clause description. The motion clips are not included; their own
licenses apply, and LAFAN1 is non-commercial. Details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
