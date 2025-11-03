import numpy as np
import pinocchio as pin

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

# === Matrice === #
def matrica_T(fi, teta, psi):

    T = np.array([
        [0, -np.sin(fi), np.cos(teta)*np.cos(fi)],
        [0, np.cos(fi), np.cos(teta)*np.sin(fi)],
        [1, 0, -np.sin(teta)]
    ])
    return T

def matrica_T_inv(fi, teta, psi):

    T_inv =  np.array([
        [(np.cos(fi)*np.sin(teta)/np.cos(teta)), (np.sin(teta)*np.sin(fi)/np.cos(teta)), 1],
        [-np.sin(fi), np.cos(fi), 0],
        [np.cos(fi)/np.cos(teta), np.sin(fi)/np.cos(teta), 0]
    ])

    return T_inv

def izvod_matrice_T(fi, teta, psi, dfi, dteta, dpsi):

    dT = np.array([
        [0,
         -np.cos(fi)*dfi,
         -np.sin(fi)*np.cos(teta)*dfi - np.cos(fi)*np.sin(teta)*dteta],

        [0,
         -np.sin(fi)*dfi,
          np.cos(fi)*np.cos(teta)*dfi - np.sin(fi)*np.sin(teta)*dteta],

        [0,
         0,
         -np.cos(teta)*dteta]
    ])
    return dT

def izvod_matriceT_inv(fi, teta, psi, dfi, dteta, dpsi):

    dT_inv = np.array([
        [(-dfi * np.sin(fi) * np.tan(teta) + dteta * np.cos(fi) / np.cos(teta) ** 2),
         (dfi * np.cos(fi) * np.tan(teta) + dteta * np.sin(fi) / np.cos(teta) ** 2),
         0],

        [-dfi * np.cos(fi),
         -dfi * np.sin(fi),
         0],

        [(-dfi * np.sin(fi) / np.cos(teta) + dteta * np.sin(teta) * np.cos(fi) / np.cos(teta) ** 2),
         (dfi * np.cos(fi) / np.cos(teta) + dteta * np.sin(teta) * np.sin(fi) / np.cos(teta) ** 2),
         0]
    ])

    return dT_inv