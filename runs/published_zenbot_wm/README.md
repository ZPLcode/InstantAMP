# Zenbot WM + AMP + PPO，I 与 J 的第 10,000 轮

两个版本均从随机权重开始。各自的完整学习器位于 `ckpt/full/snapshot.npz`，原生 MuJoCo 回放使用 `ckpt/snapshot.npz`。每个子目录的 `snapshot.json` 记录训练时长、更新次数、原始检查点编号和哈希。

| 版本 | 冷启动首次站稳 | 起身中位数 | 站起后漂移中位数 | 起身→走→跑→停 |
| --- | ---: | ---: | ---: | ---: |
| I | 55/55 | 2.82 s | 8.28 cm | 2/6 |
| J | 55/55 | 3.06 s | 12.38 cm | 4/6 |

站立、八种指令及急停的分项数据在 `i/eval` 和 `j/eval`。参考与策略的并排录像在各自的 `video` 目录。

[安装与续训](../../docs/zenbot-wm-handoff.md) · [改动历史](../../docs/zenbot-wm-history.md) · [实验方案](../../docs/zenbot-wm-plan.md)
