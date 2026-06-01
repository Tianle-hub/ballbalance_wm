# TODO

## Align Training Data With Random-IC Evaluation

- [ ] Extend `scripts/collect_dataset.py` so training datasets can sample randomized initial conditions matching `eval_rssm_random_ic_rollout.py`.
- [ ] Add CLI options such as `--init-pos-bound`, `--init-vel-bound`, and `--init-angle-bound`.
- [ ] Randomize the initial board state `theta_x, theta_y` during collection when requested.
- [ ] Keep the current default behavior unchanged for backward compatibility.
- [ ] Add tests verifying generated initial states stay within configured bounds.

Context:

Current dataset collection randomizes initial ball position and velocity using environment defaults, but starts board angles at zero:

```text
x, y       in [-0.20, 0.20]
vx, vy     in [-0.05, 0.05]
theta_x/y  = 0
```

The random-IC evaluation can test a wider distribution:

```text
x, y       in [-0.25, 0.25]
vx, vy     in [-0.20, 0.20]
theta_x/y  in [-0.12, 0.12]
```

If evaluation uses the wider distribution, rollout error may reflect out-of-distribution generalization rather than only model quality.
