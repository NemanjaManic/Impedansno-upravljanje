import numpy as np

def sinteza_trajektorije(x_start, x_goal, t, T_traj):

    x_start = np.asarray(x_start)
    x_goal = np.asarray(x_goal)

    if t <= 0:
        return x_start.copy(), np.zeros_like(x_start), np.zeros_like(x_start)
    elif t >= T_traj:
        return x_goal.copy(), np.zeros_like(x_goal), np.zeros_like(x_goal)
    else:
        T = T_traj
        x_d = (6 * (x_goal - x_start) * t ** 5 / T ** 5 - 15 * (x_goal - x_start) * t ** 4 / T ** 4 + 10 * (x_goal - x_start) * t ** 3 / T ** 3 + x_start)
        dx_d = (30 * (x_goal - x_start) * t ** 4 / T ** 5 - 60 * (x_goal - x_start) * t ** 3 / T ** 4 + 30 * (x_goal - x_start) * t ** 2 / T ** 3)
        ddx_d = (120 * (x_goal - x_start) * t ** 3 / T ** 5 - 180 * (x_goal - x_start) * t ** 2 / T ** 4 + 60 * (x_goal - x_start) * t / T ** 3)
        return x_d, dx_d, ddx_d

