# Reward Design

The non-terminal reward is shaped to stay smooth for the reward and value
models while still making the center and the board edge easy to distinguish.

Inside the board:

```python
dist2 = x**2 + y**2
center_bonus = exp(-dist2 / (2 * center_sigma**2))
edge_penalty = quadratic_warning_from_abs_position_0_4_to_0_5

reward = (
    1.0
    - center_bonus_weight
    + center_bonus_weight * center_bonus
    - pos_weight * dist2 / half_board**2
    - edge_warning_weight * edge_penalty
    - vel_weight * (vx**2 + vy**2)
    - angle_weight * (theta_x**2 + theta_y**2)
    - action_weight * (theta_x_cmd**2 + theta_y_cmd**2)
)
```

The center bonus is implemented this way so the reward is still exactly `1.0`
at the center without clipping away the useful exponential shape around it.

The edge warning starts at `|position| = 0.40`, which is 80% of the way to the
default board boundary at `0.50`. It increases quadratically until the terminal
boundary, where falling still returns `-1.0`.

Default reward parameters:

```text
reward_center_bonus_weight = 0.25
reward_center_sigma = 0.05
reward_edge_warning_start = 0.40
reward_edge_warning_weight = 0.75
```

For review, see the generated one-dimensional curve with velocity, board angle,
and action set to zero:

```text
docs/reward_position_curve.svg
```
