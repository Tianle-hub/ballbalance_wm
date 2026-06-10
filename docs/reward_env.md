Yes, I’d consider it, but I’d use it as a center bonus, not necessarily replace the whole reward with only an exponential.

Your current quadratic position term barely distinguishes 0.00 m from 0.01 m. An exponential can make “very near center” meaningfully better:

dist2 = x**2 + y**2
center_bonus = np.exp(-dist2 / (2.0 * center_sigma**2))
For example, center_sigma=0.05 means being 1 cm away still costs about 0.02 reward, much stronger than the current 0.0004.

A good shaped reward could be:

dist2 = x**2 + y**2
vel2 = vx**2 + vy**2
angle2 = theta_x**2 + theta_y**2
action2 = theta_x_cmd**2 + theta_y_cmd**2

center_bonus = np.exp(-dist2 / (2.0 * 0.05**2))

reward = (
    center_bonus
    - 0.15 * vel2
    - 1.0 * angle2
    - 0.02 * action2
)
I’d avoid making the exponential too narrow at first. If center_sigma is tiny, the reward becomes almost sparse unless the ball is already centered, and learning can get harder. Try 0.05, then maybe 0.03.

My strongest recommendation: use a hybrid reward:

broad_position = -0.5 * dist2 / (0.5**2)
center_bonus = 0.5 * np.exp(-dist2 / (2.0 * 0.05**2))

reward = (
    1.0
    + broad_position
    + center_bonus
    - vel_weight * vel2
    - angle_weight * angle2
    - action_weight * action2
)
That gives broad guidance from far away, plus a sharper reason to stop at the center. Also increase velocity penalty a bit, because oscillation is often “near center but still moving,” not just “wrong position.”