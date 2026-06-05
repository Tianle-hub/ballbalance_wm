# Reward Model Training Issue 0604

This note is historical. The old reward model failed mainly because rare fall transitions used an enormous reward
penalty, so reward MSE was dominated by outliers and MPC could overestimate impossible imagined recovery after a
fall.

The current code no longer keeps those reward modes. The active design is:

```text
normal step: bounded dense center reward in [-1, 1]
fall step:   reward = -1 and terminated = True
model:       reward regression head + separate continuation head
planning:    learned reward return is discounted by predicted continuation
```

The important remaining checks are:

- Regenerate datasets after reward changes; old datasets still contain old reward labels.
- Use `done` only to mask post-episode padding.
- Use `terminated` as the continuation target, so time-limit truncation does not become a learned failure state.
- Keep reward regression smooth; do not reintroduce large terminal reward penalties.
