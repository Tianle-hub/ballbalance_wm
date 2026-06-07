# Architecture Notes

## Modules

- `ball_rssm.models.rssm.RSSM`: Gaussian recurrent state-space model with deterministic GRU memory `h` and stochastic latent `z`.
- `ball_rssm.models.world_model.WorldModel`: encoder, RSSM, observation decoder, reward head, and continuation head.
- `ball_rssm.models.behavior.Actor`: tanh-squashed diagonal Gaussian actor over normalized action units.
- `ball_rssm.models.behavior.Critic`: scalar value model over Dreamer features `[h, z]`.
- `ball_rssm.trainer.Trainer`: combined Dreamer training loop for dynamics, actor, and critic.
- `ball_rssm.agent.DreamerAgent`: stateful environment-facing policy wrapper.

## Latent State Flow

Training data follows:

```text
obs[:, t] + action[:, t] -> obs[:, t + 1]
```

The world model encodes normalized observations and runs posterior inference over `obs[0:T+1]` with actions `action[0:T]`. Dreamer behavior learning then flattens posterior states across batch and time, subsamples up to `behavior_batch_size` starts, and launches imagined rollouts from those states.

RSSM feature convention:

```text
feature_t = concat(h_t, z_t)
```

The actor, critic, decoder, reward head, and continuation head all consume this feature.

## Losses

World model:

```text
L_world = L_reconstruction
        + beta_kl * L_KL
        + reward_loss_weight * L_reward
        + continuation_loss_weight * L_continuation
```

Actor:

```text
maximize TD(lambda) returns predicted from imagined rewards, continuation, and value
```

Critic:

```text
minimize squared error to target-critic TD(lambda) returns
```

World model parameters and critic parameters are frozen during the actor update, but gradients still flow through the differentiable RSSM transition and value function to actor actions.

## Checkpoint Contract

Dreamer checkpoints contain:

- `world_model_state_dict`
- `actor_state_dict`
- `critic_state_dict`
- `target_critic_state_dict`
- `normalizer`
- `world_model_config`
- `actor_config`
- `critic_config`
- `dreamer_config`
- optimizer states for all three optimizers

For compatibility with world-model diagnostic scripts, checkpoints also keep `model_state_dict`, `optimizer_state_dict`, and `config` aliases for the world model.

## Runtime Control

`DreamerAgent` keeps the latest RSSM belief and previous real action. At each environment step it:

1. assimilates the current observation with `posterior_update`;
2. runs the actor on the current feature `[h, z]`;
3. denormalizes and clips the action to the environment action space;
4. stores the action so the next observation can update the belief.

There is no action-sequence search at runtime.
