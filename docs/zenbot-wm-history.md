# Zenbot 起身与全方向 AMP 的改动历史

本文记录训练路线、保留下来的实现以及结果退化的尝试。文中的 I、J 专指 **11 月 23 日的联合策略实验**。早期单训起身实验用目录名标识，以免把两批实验的字母编号混在一起。

## 11 月 20–22 日，从 G1 移植到 Zenbot

| 阶段 | 做了什么 | 对后续的影响 |
| --- | --- | --- |
| G1 移植 | 把 LeggedLab 的倒地初始化、起身参考和控制流程移进 InstantAMP，保存了 G1 训练日志与策略 | 建立了可在原生 MuJoCo 中逐个倒地姿态检查的入口。[移植提交](https://github.com/ZPLcode/InstantAMP/commit/cbf7e53) · [训练记录](https://github.com/ZPLcode/InstantAMP/commit/10aaa05) |
| Zenbot 43 kg、23 自由度 | 早期尝试过把 G1 动作转给 Zenbot。随后接入了 LeggedLab `get_up_frax` 的 11 段 Zenbot 原生起身，统一关节顺序、质量和 50 Hz 参考 | 今天的倒地开局和起身 AMP 都取这 11 段。发布的旧控制器和新策略可以使用同批开局评估。[来源与哈希](../training_logs/zenbot23_getup_standing_amp_pd/recovery_sources.json) · [首批发布](https://github.com/ZPLcode/InstantAMP/commit/778f58c) |
| 起身后站歪、动作冲 | 加入站稳与位移评估，区分手撑地、转移到脚、首次站稳和随后再摔。把动作变化率的惩罚提高到原来的三倍 | 当前联合配方的 `action_rate_l2_weight` 为 `-0.015`。保留训练和回放之间相同的电机顺序、延迟与目标映射，修正了额外把 PD 目标截到关节范围的实现，实际扭矩仍受执行器限制。[物理控制](../instantamp/sim.py) · [回放与回归检查](../tests/test_zenbot.py) |
| 旧起身与站姿基线 | `002575` 所属的上游发布实验记录为 `on_policy_ppo_amp`，其配置从较早的策略初始化；仓库同时提供了通用 WM 训练器 | 早期演示证明相应旧策略能起身。它的迭代数、初始权重和观测接口要与这次从零的 WM 实验分别核对。[发布提交](https://github.com/ZPLcode/InstantAMP/commit/7c50355) · [旧实验配置](../training_logs/zenbot23_getup_standing_amp_pd/config.json) |
| 统一走跑和起身 | 做过已有走跑与起身控制器组合、教师策略蒸馏、分阶段 PPO 训练和物理参考指导的对照。随后把训练观测移到 Motion Train 的接口 | 交付的 I、J 都从随机权重开始。推理只用一个 234→23 策略；训练时的世界模型和判别器不进入部署输入。[训练观测](../instantamp/zenbot_motion_train_obs.py) · [原生策略执行](../instantamp/native_zenbot.py) |

## 先单练起身，找到影响速度和站稳的信号

早期 `zenbot_floor_amp_seed42` 的 55 个参考开局都能起身，起身时间中位数仅 0.84 s，录像里有快速蹬腿。此后除成功率外，还记录参考时长、髋关节速度、手部受力、起身位移与站稳后的再次跌倒。下表中的“原开局”保留参考起始速度，“冷启动”将起始速度清零再施加固定扰动。

| 单训起身实验，目录前缀 `zenbot_`、后缀 `_seed42` | 原开局一次站稳 / 55 | 冷启动一次站稳 / 55 | 留下的经验 |
| --- | ---: | ---: | --- |
| `floor_style` | 51 | — | AMP 可以学会起身，但仍需检查起身时间和接触顺序 |
| `timed_style`、`tracked_style` | 55、55 | 55、— | 按参考时钟奖励对应姿态能提高部分指标，起身时间仍可能偏快 |
| `paced_style`、`transition_style` | 1、0 | — | 修改相位与过渡信号曾使学到的站稳明显退化 |
| `tempo_ramp_style`、`tempo_examples_style` | 55、46 | 55、46 | 实验比较过速率课程及姿态跟踪。最终按需求将 `reference_weight` 固定为 0，保留 AMP 中的节奏负样本 |
| `amp_first_cold_style` | 2 | 1 | 全从倒地首帧开始，以 AMP 和晚期开启的抬升奖励训练，有些策略能起来但很快又跌倒 |
| `amp_first_practice_style` | 50 | 50 | 在倒地重置中抽 35% 中间支撑姿态后更容易完成从手到脚的承重转移。它成为联合训练的起点设计 |

这里的中间支撑姿态只设置仿真的初始状态。机器人随后自行施力起身。联合配方使用未配对的连续 AMP 窗口，骨盆高于 0.44 m 后才奖励向上进展；训练过程中施加的辅助抬升力为 0。早期的 `amp_first_gate_style` 只保有 1,500 轮快照，未参加上表的万轮比较。[当前配方](../tools/recipes/train_zenbot_wm_floor_amp.sh)

## 11 月 23 日，从零联合训练的 A–J

参考过 `AMP_mjlab` 的 G1 环境。它在同一任务中配置走跑与恢复数据、从动作中重置，并用 AMP PPO 更新策略。Zenbot 联合实验沿用两类开局和指令跟踪的思路，采用 InstantAMP 的世界模型更新，并为起身和直立各保留一个判别器。[参考环境](https://github.com/ccrpRepo/AMP_mjlab/blob/6c7a2947fccc973e4af8e6d90e550400f1b6fcfc/src/tasks/amp_loco/config/g1/env_cfgs.py) · [参考训练配置](https://github.com/ccrpRepo/AMP_mjlab/blob/6c7a2947fccc973e4af8e6d90e550400f1b6fcfc/src/tasks/amp_loco/config/g1/rl_cfg.py)

下表的 A–I 都取 seed 42 第 10,000 轮。每版各自初始化，一个训练进程内学习全部技能。一次站稳统计 11 个倒地开局 × 5 个固定扰动种子；漂移为站起后骨盆 XY 最大位移在试验间的中位数。连续完成要求在一次物理轨迹上完成起身、站稳、走、跑、停的各项判据。

| 版本与运行目录简称 | 相对前版的主要变化 | 冷启动一次站稳 / 55 | 起身中位数 | 漂移 | 连续完成 / 6 |
| --- | --- | ---: | ---: | ---: | ---: |
| A `floor_gaits` | 一次训练中混合倒地与直立开局，加入三轴速度指令、站立和起身 AMP | 50 | 1.82 s | 14.85 cm | 3 |
| B `progress_amp12` | 起身 AMP 从 6 调至 12；速度任务以静止时的得分为基线 | 35 | 4.38 s | 12.29 cm | 3 |
| C `command_amp` | 直立 AMP 加入当前指令，按动作类别取专家窗口。每窗 10 帧，间隔 60 ms | 34 | 4.56 s | 21.82 cm | 2 |
| D `matched_negatives` | 加入错指令的专家窗口作为负样本，改善判别器对动作类别的区分 | 30 | 4.16 s | 17.29 cm | 4 |
| E `amp_excursion` | 加入保留根运动、缩小关节摆幅的负样本，起身 AMP 调至 8 | 40 | 2.57 s | 10.73 cm | 4 |
| F `stride_velocity` | 速度用 0.24 s 平滑；调整倒地和零指令采样，起身 AMP 提到 10，想象沿用真实指令 | 40 | 3.21 s | 8.52 cm | 0 |
| G `ground_stride` | 世界模型想象从 16 步延至 28 步，直立 AMP 从 6 调至 10 | 55 | 2.96 s | 10.35 cm | 6 |
| H `source_speeds` | 同时采样名义速度和原动作速度，使转弯、侧移、后退都能匹配相应专家窗口 | 50 | 3.24 s | 12.12 cm | 5 |
| I `half_second` | 保留 H 的采样，将速度平滑从 0.24 s 调至 0.50 s | 55 | 2.82 s | 8.28 cm | 2 |
| J `upright_mirror` | 在 H 的 0.24 s 设置上，对脚部已经稳定承重的真实 PPO 样本增加系数 0.2 的镜像一致性 | 55 | 3.06 s | 12.38 cm | 4 |

完整目录名为 `zenbot_joint_<简称>_seed42`。发布标签保留 I、J 的实际学习器，A–I 的逐版汇总和报告指纹在 `runs/published_zenbot_wm/history.json`。G 的连续切换最好，H 的左右转身接近参考，但两侧表现不对称。I 在起身和站姿上更稳，左右慢速侧移和跑后突然停止都有退化。因此本次交接保留了两个独立快照。[固定发布标签](https://github.com/ZPLcode/InstantAMP/tree/zenbot-wm-handoff-20261123/runs/published_zenbot_wm)

J 的镜像映射已与 Motion Train 原实现比较，512 组观测及动作的最大绝对误差均为 0。J 的第一次启动在第 0 轮编译时遇到 JAX 条件分支统计字段不一致，补齐预热分支的 `gait_mirror_mse` 后重新从随机权重启动。A、G 也保留过编译阶段退出的启动目录。上表的成绩均来自产生了学习器更新的运行。[镜像代码](../instantamp/zenbot_motion_train_obs.py) · [完整策略更新回归测试](../tests/test_zenbot_wm_floor_amp.py)

J 最后 55/55 次都有手到脚的负载转移，站起后再次跌倒 0 次，配对急停通过 5/5。原速左右侧移的指令通过数分别为 0/5 和 4/5；对应膝摆幅为 26.8° 和 2.8°，右侧仍有碎步。两种停止对照、动作参考和录像均与第 10,000 轮的策略哈希对应。[J 的训练结果](https://github.com/ZPLcode/InstantAMP/tree/zenbot-wm-handoff-20261123/runs/published_zenbot_wm/j)

## 连续指令上的 40% 起身实验

上游随后引入连续速度指令、步态接触奖励和实际前进时切换到零指令的训练，并把联合配方扩到 4,096 个环境。新实验基于该提交，把平面速度与偏航角速度的指数跟踪权重分别从 2.0、1.5 调到 4.0、3.0。每次重置固定 1,638 个环境倒地、2,458 个环境直立；给真实 PPO 的 410 个环境中有 164 个起身环境。速度跟踪的宽度仍为 0.5，动作变化率权重采用连续指令版的 `-0.03`。[上游连续指令版](https://github.com/ZPLcode/InstantAMP/commit/fbb074365c0405142666e5e8b3ac69df54a7a11f) · [本次实验提交](https://github.com/ZPLcode/InstantAMP/commit/db242b507b40eef966dd038ada8384fb0043d654)

在同一组原生 MuJoCo 冷启动测试中，第 5,000 轮从 11 种完整倒地姿态各试 5 个扰动，站起 0/55。另按训练代码抽取开局并统一给零速度指令，原始倒地姿态站稳 0/123，参考动作中间姿态站稳 9/69。第 6,500 轮在固定种子的 11 个完整开局中一次站稳 2/11，第 7,000 轮增至 9/11。第 7,500 轮在完整的五种扰动测试中达到 55/55，全部测到手先承重再转移到脚上。起身用时中位数为 3.76 秒，站起后骨盆最大水平位移中位数为 7.0 厘米。

这条训练从随机权重开始。首个进程记录到第 6,650 轮后因主机内存不足退出。训练从本次第 6,500 轮保存的完整学习器恢复，沿用策略、世界模型、两个 AMP 判别器、价值函数和优化器状态。恢复时重新初始化仿真环境和回放缓冲。两段训练使用相同的奖励权重和环境分配。

第 10,000 轮共记录 655,884,288 次物理环境转移、30,000 次世界模型更新和 29,352 次策略更新。相同的 55 个冷启动全部起身并保持最后 2 秒安静站立，50 次一次完成，5 次在地面重试，站起后再次倒地 0 次。55 次都有手到脚的受力转移。起身时间中位数为 2.62 秒，与各自参考时长之比的中位数为 0.96。站起后骨盆最大水平位移中位数为 4.2 厘米，静站躯干倾角中位数为 0.75°。从站姿施加扰动的另 20 次测试里，3 次保持平衡，17 次倒地后都重新站稳。[冷启动报告](https://github.com/ZPLcode/InstantAMP/blob/zenbot-continuous-track2-floor40-10000-20261123/runs/zenbot_continuous_track2_floor40_seed42/cold/summary.json) · [扰动恢复报告](https://github.com/ZPLcode/InstantAMP/blob/zenbot-continuous-track2-floor40-10000-20261123/runs/zenbot_continuous_track2_floor40_seed42/fall_recovery.json)

同一个策略在参考录制速度下，站立、前走、后退和双向原地转弯各通过 5/5 次指令测试，跑步和双向侧移各通过 0/5 次。连续起身、走、跑、停的 6 条测试中，走和停各通过 6/6，跑通过 0/6。发布标签保留第 7,500 轮和第 10,000 轮的完整学习器及策略，以及第 6,500 轮的恢复点。[八类指令实测](https://github.com/ZPLcode/InstantAMP/blob/zenbot-continuous-track2-floor40-10000-20261123/runs/zenbot_continuous_track2_floor40_seed42/command_style_recorded/summary.json) · [训练记录、权重和录像](https://github.com/ZPLcode/InstantAMP/tree/zenbot-continuous-track2-floor40-10000-20261123/runs/zenbot_continuous_track2_floor40_seed42)

## I 的万轮结果，按技能拆开看

共采集 163,905,536 次物理环境转移，世界模型更新 30,000 次，策略更新 29,420 次，训练计时约 54 分钟。55/55 个冷启动一次起身并保持最后 2 s 安静站立，55/55 个记录到手撑地后向脚转移负载，起身后再次跌倒 0 次。另一组 33 个起身后的站姿测试，相对 AMP 站姿的关节角误差中位数 2.36°，躯干倾角中位数 1.03°。

以下每项从静止站姿开始，有 5 个测试种子。直线和侧向速度用 m/s，原地转弯用 rad/s。通过数衡量指令执行，膝摆幅单独衡量动作风格。策略和参考的摆幅均取自 `eval/recorded_commands.json`。

| 动作 | 参考指令 | 实际速度中位数 | 指令通过 | 膝摆幅，策略 / 参考 |
| --- | ---: | ---: | ---: | ---: |
| 站立 | 0 | 约 0 | 5/5 | 0.0° / 0.0° |
| 前走 | 1.000 | 0.929 | 5/5 | 48.5° / 44.7° |
| 跑 | 2.500 | 2.888 | 3/5 | 68.0° / 81.0° |
| 后退 | −0.664 | −0.576 | 4/5 | 49.2° / 53.0° |
| 左侧移 | 0.165 | 0.006 | 0/5 | 5.1° / 36.5° |
| 右侧移 | −0.172 | −0.003 | 0/5 | 1.3° / 32.4° |
| 原地左转 | 1.174 | 1.137 | 5/5 | 47.5° / 49.0° |
| 原地右转 | −1.326 | −1.324 | 5/5 | 51.5° / 52.9° |

用名义指令评估时，左右侧移分别通过 2/5、5/5；原始慢速下两侧都是 0/5。单独的急停对照中，从同一跑动状态直接给零指令有 3/5 成功，2 次跌倒；先走 1 s 再给零指令为 5/5。交接时将慢速侧移与急停列为独立验收项。[I 的原始分项报告与视频](https://github.com/ZPLcode/InstantAMP/tree/zenbot-wm-handoff-20261123/runs/published_zenbot_wm/i)

## 这次为换机做的整理

- 在同一从零配方中显式命名 I，并保留已有 J 入口。续训入口直接读保存的配置、学习器和配套策略，新增对旧 I 配置、错误文件及轮数的测试。[从零入口](../tools/recipes/train_zenbot_wm_floor_amp.sh) · [恢复入口](../tools/resume_zenbot_wm.py) · [测试](../tests/test_zenbot_wm_resume.py)
- 修复完整检查点目录在另一块盘上时的源配置定位。续训重新采样并填充缓冲区，因此学习器参数可以接续，后续轨迹会变化。[恢复与采样](../instantamp/train.py)
- 以真实运行保存的哈希锁定 11 段倒地、17 段步态和 1 份采样配置。训练和数值验收支持无外观网格的机器；桌面回放和视频仍可使用原模型网格。[数据准备](../tools/recipes/prepare_zenbot_wm_data.sh) · [校验清单](../assets/zenbot_wm_reference.sha256)
- 发布每个版本的一套完整学习器、配套 actor、训练配置、逐次日志、训练时源码和评估。训练缓存、随机失效的临时路径及未训练的候选修改都不作为换机入口。[交接步骤](zenbot-wm-handoff.md)
