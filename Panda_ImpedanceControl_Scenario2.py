#OVA SIMULACIJA PREDSTAVLJA DRUGI SCENARIO

import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer
from robot_descriptions.loaders.pinocchio import load_robot_description as load_pinocchio
from robot_descriptions.loaders.mujoco import load_robot_description as load_mujoco
from Funkcije import sinteza_trajektorije
import matplotlib.pyplot as plt

# === Učitavanje modela === #
# ---pinocchio model--- #
pinocchio_robot = load_pinocchio("panda_description")
pinocchio_model = pinocchio_robot.model
pinocchio_data = pinocchio_robot.data
# ---mujoco model--- #
mujoco_model = load_mujoco("panda_mj_description")
mujoco_data = mujoco.MjData(mujoco_model)

# === Parametri simulacije === #
dt = 0.001
T = 30.0
N = int(T/dt)
mujoco_model.opt.timestep = dt

# --- Pocetne konfiguracije --- #
q0 = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
dq0 = np.zeros(pinocchio_model.nv)
mujoco_data.qpos[:] = q0.copy()
mujoco_data.qvel[:] = dq0.copy()
v = viewer.launch_passive(mujoco_model, mujoco_data)

# ===TCP alata(hvataljke) u urdf fajlu - koordinatni sistem na kraju effektora=== #
frame = "panda_hand_tcp"
frame_id = pinocchio_model.getFrameId(frame)

# === Parametri impedanse === #
#--rucno--
#kruta pozicija, meka orijentacija
# Km = np.diag([14000, 14000, 14000, 90, 90, 90])
# Hm = np.diag([50, 50, 50, 1, 1, 1])
# Dm = np.diag([2000, 2000, 2000, 12.5, 12.5, 12.5])
#kruta orijentacija, meka pozicija
# Km = np.diag([12000, 12000, 12000, 180, 180, 180])
# Hm = np.diag([150, 150, 150, 0.3, 0.3, 0.3])
# Dm = np.diag([2000, 2000, 2000, 12.5, 12.5, 12.5])
#kruta orijentacija, kruta z, x i y meke
Km = np.diag([12000, 12000, 14000, 180, 180, 180])
Hm = np.diag([150, 150, 50, 0.3, 0.3, 0.3])
Dm = np.diag([2000, 2000, 2000, 12.5, 12.5, 12.5])
#pozicija i orijentacija
x_d = np.array([0.4, 0.0, 0.4])
rpy_const = np.array([-np.pi, 0.0, 0.0])
dx_d = np.zeros(6)
ddx_d = np.zeros(6)
Fext = np.zeros(6)

# === Liste za prikupljanje podataka === #
q_history, dq_history, tau_history = [], [], []
tcp_pos_history, Fext_history, time_history = [], [], []

# === Glavna simulacija - tok simulacije === #
for i in range(N):
    t = i * dt #vreme
    #stanje iz mujoca
    q = mujoco_data.qpos[:].copy()
    dq = mujoco_data.qvel[:].copy()
    mujoco.mj_forward(mujoco_model, mujoco_data)

    # Podaci o end effectoru iz Pinocchia
    pin.framesForwardKinematics(pinocchio_model, pinocchio_data, q)
    tcp_frame = pinocchio_data.oMf[frame_id] #matrica homogene transformacije
    tcp_pos = tcp_frame.translation.copy() #translacija
    tcp_rot = tcp_frame.rotation.copy() #rotacija

    if 2.5 < t < 4.5:
        Fext = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # sila po X
    elif 6.5 < t < 8.5:
        Fext = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])  # sila po Y
    elif 10.5 < t < 12.5:
        Fext = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # sila po Z
    elif 14.5 < t < 16.5:
        Fext = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    elif 18.5 < t < 20.5:
        Fext = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    elif 22.5 < t < 24.5:
        Fext = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    else:
        Fext = np.zeros(6)

    #Jakobijani
    #geometrijski Jakobijan
    J = pin.computeFrameJacobian(pinocchio_model, pinocchio_data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #izvod geometrijskog Jakobijana
    dJ = pin.frameJacobianTimeVariation(pinocchio_model, pinocchio_data, q, dq, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #pseudoinverzni Jakobijan
    U, s, Vt = np.linalg.svd(J)
    sigma_min = np.min(s)
    epsilon = 0.03
    k_max = 0.24
    k_squared = 0.0 if sigma_min >= epsilon else (1 - (sigma_min / epsilon)**2) * k_max**2
    pInvJ = J.T @ np.linalg.inv(J @ J.T + k_squared * np.eye(6))
    dx = J @ dq


    r_d = pin.rpy.rpyToMatrix(rpy_const)
    # ---poziciona greska---
    pos_error = x_d[:3] - tcp_pos
    # ---greska orijentacije---
    rot_err_local = pin.log3(tcp_rot.T@r_d)
    rot_err_world = tcp_rot @ rot_err_local
    x_error = np.concatenate([pos_error, rot_err_world])
    dx_error = dx_d - dx

    H = pinocchio_robot.mass(q)
    h = pinocchio_robot.nle(q,dq)
    deo1 = H @ pInvJ @ (ddx_d - dJ @ dq + np.linalg.inv(Hm) @ (Dm@dx_error + Km@x_error))
    deo4 = (H @ pInvJ @ np.linalg.inv(Hm) - J.T) @ Fext
    tau = deo1 + h + deo4

    # NUL-PROSTOR === #
    q_center = 0.5 * (pinocchio_model.lowerPositionLimit + pinocchio_model.upperPositionLimit)
    q_pref = q_center.copy()
    Kp_null = np.diag([10, 10, 10, 8, 8, 8, 5, 0, 0])
    Dp_null = np.diag([2, 2, 2, 1.6, 1.6, 1.6, 1.2, 0, 0])
    Hx = np.linalg.inv(J @ np.linalg.inv(H) @ J.T)
    J_null = np.linalg.inv(H) @ J.T @ Hx
    N = np.eye(pinocchio_model.nv) - J_null @ J
    tau_null = N @ (
                Kp_null @ (q_pref[:pinocchio_model.nv] - q[:pinocchio_model.nv]) - Dp_null @ dq[:pinocchio_model.nv])

    tau = tau + tau_null
    # =============================== #

    mujoco_data.ctrl[:] = np.concatenate([tau[:7], [0.0]])
    mujoco_data.ctrl[:] = np.concatenate([tau[:7], [0.0]])
    mujoco.mj_step(mujoco_model, mujoco_data)
    v.sync()

    # Snimanje podataka
    q_history.append(q.copy())
    dq_history.append(dq.copy())
    tau_history.append(tau.copy())
    tcp_pos_history.append(tcp_pos.copy())
    Fext_history.append(Fext.copy())
    time_history.append(t)

# === Pretvaranje u numpy nizove === #
q_history = np.array(q_history)
dq_history = np.array(dq_history)
tau_history = np.array(tau_history)
tcp_pos_history = np.array(tcp_pos_history)
Fext_history = np.array(Fext_history)
time_history = np.array(time_history)

# === Izračunaj orijentaciju TCP-a=== #
rpy_history = []
for q in q_history:
    pin.framesForwardKinematics(pinocchio_model, pinocchio_data, q)
    tcp_frame = pinocchio_data.oMf[frame_id]
    tcp_rot = tcp_frame.rotation
    rpy = pin.rpy.matrixToRpy(tcp_rot)
    rpy_history.append(rpy)
rpy_history = np.array(rpy_history)

# === Crtanje rezultata === #
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 9,
})

plt.figure(figsize=(12, 12))

# --- Kontaktne sile i momenti ---
plt.subplot(2, 1, 1)
plt.plot(time_history, Fext_history[:, 0], color='tab:red', label='Fx [N]')
plt.plot(time_history, Fext_history[:, 1], color='tab:blue', label='Fy [N]')
plt.plot(time_history, Fext_history[:, 2], color='gray', label='Fz [N]')
plt.plot(time_history, Fext_history[:, 3], '--', color='tab:red', label='Mx [Nm]')
plt.plot(time_history, Fext_history[:, 4], '--', color='tab:blue', label='My [Nm]')
plt.plot(time_history, Fext_history[:, 5], '--', color='gray', label='Mz [Nm]')
plt.xlabel('Vreme [s]')
plt.ylabel('Sile i momenti')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
# --- Pozicije zglobova ---
plt.subplot(2, 1, 2)
for j in range(q_history.shape[1]):
    plt.plot(time_history, q_history[:, j], label=f'q{j+1}')
plt.xlabel('Vreme [s]')
plt.ylabel('Pozicije u zglobovima [rad]')
plt.legend(ncol=4, fontsize=8, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.08, hspace=0.4)
plt.show()

# === Druga figura: Kontaktne sile + pozicija TCP-a + orijentacija TCP-a === #
plt.figure(figsize=(12, 14))

# --- Kontaktne sile i momenti ---
plt.subplot(3, 1, 1)
plt.plot(time_history, Fext_history[:, 0], color='tab:red', label='Fx [N]')
plt.plot(time_history, Fext_history[:, 1], color='tab:blue', label='Fy [N]')
plt.plot(time_history, Fext_history[:, 2], color='gray', label='Fz [N]')
plt.plot(time_history, Fext_history[:, 3], '--', color='tab:red', label='Mx [Nm]')
plt.plot(time_history, Fext_history[:, 4], '--', color='tab:blue', label='My [Nm]')
plt.plot(time_history, Fext_history[:, 5], '--', color='gray', label='Mz [Nm]')
plt.xlabel('Vreme [s]')
plt.ylabel('Sile i momenti')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
# --- Pozicija TCP-a ---
plt.subplot(3, 1, 2)
plt.plot(time_history, tcp_pos_history[:, 0], color='tab:red', label='x [m]')
plt.plot(time_history, tcp_pos_history[:, 1], color='tab:blue', label='y [m]')
plt.plot(time_history, tcp_pos_history[:, 2], color='gray', label='z [m]')
plt.xlabel('Vreme [s]')
plt.ylabel('Pozicija TCP-a [m]')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
# --- Orijentacija TCP-a ---
plt.subplot(3, 1, 3)
rpy_unwrapped = np.unwrap(rpy_history, axis=0)
plt.plot(time_history, rpy_unwrapped[:, 0], color='tab:red', label='Roll [rad]')
plt.plot(time_history, rpy_unwrapped[:, 1], color='tab:blue', label='Pitch [rad]')
plt.plot(time_history, rpy_unwrapped[:, 2], color='gray', label='Yaw [rad]')
plt.xlabel('Vreme [s]')
plt.ylabel('Orijentacija TCP-a [rad]')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.08, hspace=0.4)
plt.show()

plt.figure(figsize=(12, 16))

