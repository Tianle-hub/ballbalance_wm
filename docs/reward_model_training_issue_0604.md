# Reward Model Training Issue 0604

## 结论

当前 `runs/rssm_ball_v2_long_wiz_reward_model` 的主要问题不是模型太简单，也不是环境 reward 公式写错。
更大的问题是 reward loss 被少量掉球惩罚样本主导，同时旧训练代码没有用 `done` mask 排除 episode 结束后的 padding transition。

## 检查结果

- 数据集是 `data/ball_balance_mpc_v1_reward_normalized.npz`。
- 数据采集模式是 `mpc_cover`，不是普通 `mixed`。
- `mpc_cover` 仍然是手写策略混合，包括 `pd`、`pd_offset`、`pd_noisy`、`random_smooth`、`random_burst`、`random_uniform`。
- 环境 reward 是默认中心稳定 reward：位置、速度、角度、动作代价之和取负，掉球时额外 `-100`。
- 用 `next_obs` 和 `action` 重新计算 reward，与数据集里保存的 reward 在有效 transition 上基本一致，平均绝对误差约 `1e-8`。
- 文件名里有 `reward_normalized`，但数据里保存的是原始环境 reward；训练时由 normalizer 标准化。

当前 checkpoint 的 reward normalizer：

```text
reward_mean = -0.3468
reward_std  =  3.4107
```

## 关键数据分布

```text
train 掉球 episode 比例: 32.2%
val 掉球 episode 比例:   37.0%

train done 后 padding step 比例: 19.1%
val done 后 padding step 比例:   21.7%
```

掉球 reward 约为 `-103`，标准化后约为 `-30`。如果模型把一个掉球 transition 预测成普通的接近 0 的 reward，
单个样本 normalized MSE 约为：

```text
30^2 ~= 900
```

validation split 中，掉球 transition 如果预测不好，单独就可以贡献约 `1.12` normalized MSE。这几乎等于最终看到的
`val/reward_loss ~= 1.13`。

## 代码问题

旧版 `WorldModel.loss()` 直接忽略了 `done_seq`：

```python
del done_seq
```

这会导致两类问题：

- post-done padding reward 是人工填充的 `0.0`，但被当成真实 transition 训练。
- 掉球终止、普通非终止 transition、padding transition 的 reward loss 混在一起，TensorBoard 上看不到真正的问题来源。

## 已加入的训练修复

训练 loss 现在使用 transition mask：

- `effective`: episode 结束前和首个 done transition。
- `nonterminal`: 有效且没有 done 的普通 transition。
- `fall_terminal`: 有效且 `terminated=True` 的掉球终止 transition。
- `post_done_padding`: episode 结束后的 padding transition。

训练目标会排除 `post_done_padding`。日志会额外记录：

```text
reward_loss_nonterminal
reward_loss_fall_terminal
reward_loss_post_done_padding
reward_effective_fraction
reward_fall_terminal_fraction
reward_post_done_padding_fraction
```

## Reward 训练模式

新增 `--reward-prediction-mode`：

```text
raw        原始 reward 回归。会使用 done mask，但仍然直接学习 -100 掉球惩罚。
clip       对标准化 reward target 做下限截断，降低掉球大惩罚对 MSE 的支配。
split_fall 连续 reward head 学习截断后的连续 reward，同时额外训练 fall probability head。
```

推荐先跑一个不覆盖旧 run 的新实验：

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_mpc_v1_reward_normalized.npz \
  --run-dir runs/rssm_ball_v2_reward_masked_raw \
  --seq-len 200 \
  --batch-size 256 \
  --epochs 100 \
  --reward-prediction-mode raw \
  --no-resume
```

如果 `reward_loss_fall_terminal` 仍然支配 validation loss，再试截断模式：

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_mpc_v1_reward_normalized.npz \
  --run-dir runs/rssm_ball_v2_reward_clip \
  --seq-len 200 \
  --batch-size 256 \
  --epochs 100 \
  --reward-prediction-mode clip \
  --reward-clip-min -5.0 \
  --fall-loss-weight 0.25 \
  --no-resume
```

如果需要显式预测掉球风险，再试 split 模式：

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_mpc_v1_reward_normalized.npz \
  --run-dir runs/rssm_ball_v2_reward_split_fall \
  --seq-len 200 \
  --batch-size 256 \
  --epochs 100 \
  --reward-prediction-mode split_fall \
  --reward-clip-min -5.0 \
  --fall-reward-threshold -10.0 \
  --fall-penalty-value -30.0 \
  --fall-loss-weight 1.0 \
  --fall-prediction-loss-weight 1.0 \
  --no-resume
```

这些阈值都是标准化 reward 空间中的值。当前数据里掉球 reward 标准化后约为 `-30`，所以：

- `--reward-clip-min -5.0` 表示连续 reward head 不直接拟合 `-30` 的掉球尖峰。
- `--fall-reward-threshold -10.0` 可作为没有 `terminated` 字段时的掉球近似阈值；当前数据有 `terminated`，会优先使用它。
- `--fall-penalty-value -30.0` 用于 `split_fall` 模式下把 fall probability 转成期望 reward 惩罚。

## 是否需要更平衡地采集边界/掉球样本

需要，但建议分两步做。

第一步先不要急着补采，先保证评估公平：

- 用 stratified split，让 train/val 的掉球 episode 比例接近。
- 分别报告非终止、掉球终止、padding 的 reward loss。
- 用 `best.pt` 做 MPC/eval，不要用过拟合后的 `latest.pt`。

第二步再补采更有诊断价值的数据：

- 增加接近边界但不掉球的 episode，让模型学会高风险区域的连续 reward。
- 增加刚好掉球的 boundary episode，让 fall head 学掉球边界。
- 控制掉球比例，不要让数据集大部分都是掉球，也不要让掉球过少。可以先尝试 train/val 都保持约 `30% - 40%` 掉球 episode。
- 如果目标是中心稳定，继续使用中心 reward；如果目标是 via-point，reward model 需要加入目标信息，否则 learned reward 天然会偏向中心。

具体做法：

```bash
# 当前覆盖型数据，保留
python scripts/collect_dataset.py \
  --num-episodes 5000 \
  --max-episode-steps 300 \
  --mode mpc_cover \
  --pos-bound 0.25 \
  --vel-bound 0.20 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --seed 0 \
  --out data/ball_balance_mpc_v1_reward_normalized.npz
```

然后新增一个专门的边界采集配方，而不是直接替换旧数据：

```bash
python scripts/collect_dataset.py \
  --num-episodes 2000 \
  --max-episode-steps 300 \
  --mode mpc_cover \
  --pos-bound 0.35 \
  --vel-bound 0.30 \
  --angle-bound 0.18 \
  --target-bound 0.20 \
  --action-noise-std 0.06 \
  --seed 10 \
  --out data/ball_balance_boundary_probe_v1.npz
```

之后检查两个数据集的掉球率、return 分布、reward 分布，再决定是合并、重采样，还是只用于 stress test。
