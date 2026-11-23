# Zenbot 43 kg 联合策略换机手册

从 I 版继续训练，使用发布目录 `runs/published_zenbot_wm/i`。这个策略在一次从随机权重开始的 10,000 轮训练中学会了倒地起身、站立和多方向运动。原始动作速度下的左右侧移和跑后急停仍需改善。已验证的结果和每一版的改动见[改动历史](zenbot-wm-history.md)，后续实验顺序见[实验方案](zenbot-wm-plan.md)。

本页的 I、J 复现命令以固定标签的源码为准。`feat/g1-amp-getup` 后来加入了连续速度指令；该分支当前的联合配方采用 4,096 个环境、40% 固定起身环境和双倍跟踪奖励。要跑这一版，用该分支的 `--locomotion` 入口。[连续指令与起停改动](https://github.com/ZPLcode/InstantAMP/commit/fbb074365c0405142666e5e8b3ac69df54a7a11f) · [40% 起身与双倍跟踪](https://github.com/ZPLcode/InstantAMP/commit/db242b507b40eef966dd038ada8384fb0043d654)

## 在新机器上取代码和权重

代码分支是 `feat/g1-amp-getup`。固定标签 `zenbot-wm-handoff-20261123` 同时包含交接时的代码、I 和 J 的完整学习器、推理策略、训练配置、评估和录像。用标签克隆后创建自己的工作分支，后续提交就能从同一个起点追溯。

```bash
git clone --depth 1 --branch zenbot-wm-handoff-20261123 https://github.com/ZPLcode/InstantAMP.git
cd InstantAMP
git switch -c work/zenbot-wm
(cd runs/published_zenbot_wm && sha256sum -c SHA256SUMS)
```

已有代码分支的机器可在仓库根目录单独取结果。下面的命令把文件写到 `runs/published_zenbot_wm`，请用空的目标目录。

```bash
git fetch --depth 1 origin tag zenbot-wm-handoff-20261123
git archive zenbot-wm-handoff-20261123 runs/published_zenbot_wm | tar -x
(cd runs/published_zenbot_wm && sha256sum -c SHA256SUMS)
```

## 安装和准备动作

实测环境是 Linux、Python 3.11、RTX 5060 Ti 16 GB 和 NVIDIA 580 驱动。训练使用 JAX CUDA 和 MuJoCo Warp，回放和评分可以在 CPU 上运行。以下命令在仓库根目录执行。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash tools/recipes/prepare_zenbot_wm_data.sh
.venv/bin/python -c 'import jax; print(jax.devices())'
```

物理模型和 11 段起身参考已在 InstantAMP 中。准备脚本用本机的 GitHub SSH 身份从 `ZPLcode/h1_train` 固定提交取出 17 段步态与站立参考，转换到 50 Hz，连同采样权重共校验 29 个文件。如果已有对应的 Motion Train 源码，可以直接传它的路径。需要走 HTTPS 的机器可用 `MOTION_TRAIN_URL` 指定该仓库的 HTTPS 地址。[源提交](https://github.com/ZPLcode/h1_train/tree/aafe0101ade37656f5390a7320bbd026f1f3c171) · [参考文件 SHA-256](../assets/zenbot_wm_reference.sha256)

```bash
bash tools/recipes/prepare_zenbot_wm_data.sh /path/to/motion_train
```

## 从 I 的第 10,000 轮接着训

```bash
unset JAX_PLATFORMS
.venv/bin/python -m tools.resume_zenbot_wm \
    runs/published_zenbot_wm/i/ckpt/full/snapshot.npz \
    runs/zenbot_i_to_12000 --until 12000
```

`--until 12000` 表示总轮数达到 12,000，再更新 2,000 轮。用新的输出目录，原始快照可以反复作为起点。J 的续训入口相同，把目录 `i` 换成 `j`。两版的实际起点以各自的 `ckpt/snapshot.json` 中的 `it` 为准。

| 发布目录中的文件 | 用途 |
| --- | --- |
| `config.json`、`initialization.json` | 还原源实验超参数，核对随机种子、初始权重指纹和策略接口 |
| `ckpt/full/snapshot.npz` | 继续更新。包含实时及平均 actor、世界模型、两个 AMP 判别器、critic、优化器、归一化统计和计数 |
| `ckpt/snapshot.npz`、`ckpt/snapshot.json` | 在 MuJoCo 运行策略，标识保存轮数。续训也会读取这份配套策略和元数据 |
| `eval/`、`video/` | 原始评估指标和参考动作与策略的并排录像。`eval/cold.json` 中有逐样本结果 |
| `training_source.tar.gz`、`log.jsonl` | 训练时使用的源码快照和逐次记录，可与整理后的入口核对 |

跨机恢复会重新初始化并填充真实采样缓冲区，重置仿真环境和随机流。学习器从保存的轮数和参数继续，后面的采样轨迹会与原机连续运行有差异。新目录的 `ckpt/0000.json` 可能对应 10,500 等绝对轮数，选模型时请读 JSON 中的 `it`。

历史 `config.json` 和评估记录保留了原机写入的路径。续训命令会替换输出目录、编译缓存和初始化路径；核对历史报告时可用其中的策略 SHA-256 对应发布文件。

## 从零复现

两条命令各自从随机权重启动一个 seed 42 的 10,000 轮实验。选其中一个执行。I 的速度平滑时间常数为 0.50 s，J 为 0.24 s 并在双脚支撑稳定后的真实 PPO 样本上启用 0.2 镜像一致性。

```bash
bash tools/recipes/train_zenbot_wm_floor_amp.sh runs/zenbot_i_seed42 --locomotion-i
```

```bash
bash tools/recipes/train_zenbot_wm_floor_amp.sh runs/zenbot_j_seed42 --locomotion
```

训练每 500 轮保存一次完整学习器和策略。以这里保存的模型大小估算，10,000 轮仅检查点约需 1 GB 空间，另留 JAX 编译缓存和依赖安装空间。训练结束会自动运行起身、各方向运动与连续指令测试。装有 Zenbot 外观网格和 `ffmpeg` 时，方向测试也会生成视频。

## 在原生 MuJoCo 上验收

```bash
policy=runs/published_zenbot_wm/i/ckpt/snapshot.npz
JAX_PLATFORMS=cpu .venv/bin/python -m tools.evaluate_zenbot_getup \
    --checkpoint "$policy" --out runs/check_i/cold --summary --cold-start --openings-only \
    --seconds 12 --reference data/zenbot23_walk_run_recovery/recovery \
    --seeds 932277 1654679 2718869 3950477 5969183
JAX_PLATFORMS=cpu .venv/bin/python -m tools.evaluate_zenbot_command_style \
    "$policy" runs/check_i/recorded --recorded-speeds
JAX_PLATFORMS=cpu .venv/bin/python -m tools.evaluate_zenbot_command_style \
    "$policy" runs/check_i/nominal
JAX_PLATFORMS=cpu .venv/bin/python -m tools.evaluate_zenbot_four_skills \
    "$policy" runs/check_i/sequence --cold-start \
    --stand-reference data/zenbot23_walk_run_recovery/walk_run/stand_4s_120f.npz \
    --seeds 932277 1654679 2718869
```

要复核手到脚的承重转移、或观察跑后直接停和先走再停的区别，可用发布标签里的诊断脚本。第一个命令读取上一步产生的逐试验物理记录；第二个命令让两种停法使用相同的起跑状态和电机延迟。

```bash
diag=runs/published_zenbot_wm/diagnostics
.venv/bin/python "$diag/ground_support.py" runs/check_i/cold runs/check_i/support.json
PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python "$diag/paired_stops.py" \
    runs/published_zenbot_wm/i snapshot runs/check_i/paired_stops.json
```

要交互观察，可从同一份 Motion Train 源码复制 Zenbot 的外观网格，在带桌面的机器打开窗口。空格播放或暂停，`N` 换倒地姿态，`R` 重置，回车施加推扰。

```bash
mkdir -p data/zenbot23_getup/meshes
cp -R /path/to/motion_train/robolab/data/robots/roboparty/zenbot/meshes/. data/zenbot23_getup/meshes/
JAX_PLATFORMS=cpu .venv/bin/python -m tools.play_zenbot_getup --checkpoint "$policy" --trial 0
```

## 从哪里读代码

| 要改的行为 | 代码入口 |
| --- | --- |
| 从零训练参数、训练结束的评测顺序 | [联合训练脚本](../tools/recipes/train_zenbot_wm_floor_amp.sh) |
| 真实交互、倒地初始化、手脚支撑、站稳后的速度任务 | [Sim](../instantamp/sim.py) 和 [Zenbot 关节与奖励](../instantamp/zenbot.py) |
| 234 维观测的时序、关节顺序、镜像映射 | [Motion Train 观测](../instantamp/zenbot_motion_train_obs.py) |
| AMP 专家窗口、指令和原始速度 | [步态采样](../instantamp/zenbot_gait_demos.py) 和 [判别器](../instantamp/amp.py) |
| 世界模型拟合、想象轨迹、真实 PPO、两种运动先验的更新 | [InstantAMP](../instantamp/train.py) |
| 原生物理回放和方向风格指标 | [ZenbotNative](../instantamp/native_zenbot.py) 和 [方向评估](../tools/evaluate_zenbot_command_style.py) |
