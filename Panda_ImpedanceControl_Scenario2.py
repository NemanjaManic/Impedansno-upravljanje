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
mujoco_model = load_mujoco("panda_mj_description_box")
mujoco_data = mujoco.MjData(mujoco_model)

# === Parametri simulacije === #
dt = 0.001
T = 15.0
N = int(T/dt)
mujoco_model.opt.timestep = dt

# --- Pocetne konfiguracije --- #
q0 = np.zeros(pinocchio_model.nq)
dq0 = np.zeros(pinocchio_model.nv)
mujoco_data.qpos[:] = q0.copy()
mujoco_data.qvel[:] = dq0.copy()
v = viewer.launch_passive(mujoco_model, mujoco_data)

# ===TCP alata(hvataljke) u urdf fajlu - koordinatni sistem na kraju effektora=== #
frame = "panda_hand_tcp"
frame_id = pinocchio_model.getFrameId(frame)

# === Parametri impedanse === #
# K_trans = np.array([10000, 10000, 500])
# K_rot   = np.array([100, 100, 100])
# K_diag  = np.concatenate([K_trans, K_rot])
# Km = np.diag(K_diag)
# wn_trans = np.array([10., 10., 1.])
# wn_rot   = np.array([16., 16., 16.])
# wn_diag  = np.concatenate([wn_trans, wn_rot])
# Hm_diag = K_diag / (wn_diag**2)
# print(Hm_diag)
# Hm = np.diag(Hm_diag)
# zeta = 1.0
# Dm_diag = 2.0 * zeta * (K_diag / wn_diag)
# Dm = np.diag(Dm_diag)

#--rucno--
Km = np.diag([10000, 10000, 3500, 100, 100, 100])
Hm = np.diag([100, 100, 380, 0.390625, 0.390625, 0.390625])
Dm = np.diag([2000, 2000, 800, 12.5, 12.5, 12.5])

#pozicija i orijentacija
x_d = np.array([0.7, 0.0, 0.4])
rpy_const = np.array([np.pi, 0.0, 0.0])
dx_d = np.zeros(6)
ddx_d = np.zeros(6)
Fext = np.zeros(6)

# === Liste za prikupljanje podataka === #
q_history, dq_history, tau_history = [], [], []
tcp_pos_history, Fext_history, time_history = [], [], []


# === Pronadji indekse senzora po imenu === #
force_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "finger_tip_force")
torque_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "finger_tip_torque")
#ukoliko je senzor izmedju prirubnice i hvataljke - koristi drugi .xml
# force_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
# torque_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
# ---dobavi adrese u sensordata nizu--- #
force_addr = mujoco_model.sensor_adr[force_sensor_id]
torque_addr = mujoco_model.sensor_adr[torque_sensor_id]

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

    if 3.0 < t < 3.2:
        Fext = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # sila po X
    elif 7.0 < t < 7.2:
        Fext = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])  # sila po Y
    elif 12.0 < t < 12.2:
        Fext = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # sila po Z
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
    k_max = 0.3
    k_squared = 0.0 if sigma_min >= epsilon else (1 - (sigma_min / epsilon)**2) * k_max**2
    pInvJ = J.T @ np.linalg.inv(J @ J.T + k_squared * np.eye(6))
    dx = J @ dq

    #
    r_d = pin.rpy.rpyToMatrix(rpy_const)
    # ---pozicioni greska---
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


plt.figure(figsize=(12,8))

# --- Kontaktne sile ---
plt.subplot(2,1,1)  # 2 reda, 1 kolona, prvi subplot
plt.plot(time_history, Fext_history[:,0], label='Fx')
plt.plot(time_history, Fext_history[:,1], label='Fy')
plt.plot(time_history, Fext_history[:,2], label='Fz')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Kontaktne sile na TCP')
plt.legend()
plt.grid(True)

# --- Pozicije zglobova ---
plt.subplot(2,1,2)  # 2 reda, 1 kolona, drugi subplot
for j in range(q_history.shape[1]):
    plt.plot(time_history, q_history[:,j], label=f'Joint {j+1}')
plt.xlabel('Time [s]')
plt.ylabel('Joint angles [rad]')
plt.title('Pozicije zglobova')
plt.legend()
plt.grid(True)

plt.tight_layout()  # lepo rasporedi subplots
plt.show()
