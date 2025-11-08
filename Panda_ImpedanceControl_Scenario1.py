#OVA SIMULACIJA PREDSTAVLJA PRVI SCENARIO

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
T = 10.0
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
K_trans = np.array([10000, 10000, 7900])
K_rot   = np.array([500, 500, 500])
K_diag  = np.concatenate([K_trans, K_rot])
Km = np.diag(K_diag)

wn_trans = np.array([10., 10., 1.8])
wn_rot   = np.array([10., 10., 10.])

wn_diag  = np.concatenate([wn_trans, wn_rot])
Hm_diag = K_diag / (wn_diag**2)
Hm = np.diag(Hm_diag)

zeta = 1.2
Dm_diag = 2.0 * zeta * (K_diag / wn_diag)
Dm = np.diag(Dm_diag)

z_const = 0.2
rpy_const = np.array([0.0, np.pi, 0.0])
A = np.array([0.55, -0.2, z_const])
B = np.array([0.65, -0.2, z_const])
C = np.array([0.65, 0.2, z_const])
D = np.array([0.55, 0.2, z_const])
waypoints = [A, D, C, B, A]
waypoints_one_loop = [A, D, C, B, A]
waypoints = waypoints_one_loop + waypoints_one_loop[1:]  # da ne duplira A na spoju
num_segments = len(waypoints) - 1
T_traj_total = 20.0
T_side = T_traj_total / num_segments
T = T_traj_total
N = int(T / dt)

# === Liste za prikupljanje podataka === #
q_history, dq_history, tau_history,k_history,Fext_raw_history = [], [], [], [], []
tcp_pos_history, Fext_history, time_history,Fext_comp_history = [], [], [],[]

# === Generisanje idealne putanje === #
num_points_per_segment = 100

# --- 1. Putanja od home pozicije do prve tačke A ---
pin.framesForwardKinematics(pinocchio_model, pinocchio_data, q0)
home_tcp = pinocchio_data.oMf[frame_id].translation.copy()

T_home_to_A = 2.0  # trajanje u sekundama
num_points_home = 100
home_to_A_path = []
segment_durations = [T_home_to_A] + [T_side]*num_segments

def get_segment_index_and_local_time(t, segment_durations):
    cum_time = 0.0
    for i, dur in enumerate(segment_durations):
        if t < cum_time + dur:
            t_seg = t - cum_time
            return i, t_seg
        cum_time += dur
    return len(segment_durations)-1, segment_durations[-1]

for t_step in np.linspace(0, T_home_to_A, num_points_home):
    x_d_pos, _, _ = sinteza_trajektorije(home_tcp, A, t_step, T_home_to_A)
    home_to_A_path.append(x_d_pos)

home_to_A_path = np.array(home_to_A_path)

# --- 2. Putanja pravougaonika ---
ideal_path_rect = []

for i in range(num_segments):
    x_start_seg = waypoints[i]
    x_goal_seg = waypoints[i+1]
    for t_step in np.linspace(0, T_side, num_points_per_segment):
        x_d_pos, _, _ = sinteza_trajektorije(x_start_seg, x_goal_seg, t_step, T_side)
        ideal_path_rect.append(x_d_pos)

ideal_path_rect = np.array(ideal_path_rect)

# --- 3. Spoji home → A sa pravougaonom ---
ideal_path = np.vstack([home_to_A_path, ideal_path_rect])

# === Pronadji indekse senzora po imenu === #
force_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
torque_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
# ---dobavi adrese u sensordata nizu--- #
force_addr = mujoco_model.sensor_adr[force_sensor_id]
torque_addr = mujoco_model.sensor_adr[torque_sensor_id]
Fext1 = np.zeros(6)

window_size = 50  # broj uzoraka za filtriranje
Fext_window = []  # lista 6D vektora    # lista koja drži poslednjih 'window_size' uzoraka sile

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

    # Dobavi frame-ove
    frame_hand_id = pinocchio_model.getFrameId("panda_hand")

    # Masa i COM
    m_total = 0.76 # hand + 2 prsta
    com_hand_local = np.array([-0.01, 0.0, 0.03])
    com_finger_local = np.array([0.0, 0.0, 0.0584])
    com_total_hand_local = (0.73 * com_hand_local + 0.015 * com_finger_local + 0.015 * com_finger_local) / m_total

    # Pozicija COM-a u svetu
    oM_hand = pinocchio_data.oMf[frame_hand_id]
    p_sensor_world = oM_hand.translation + oM_hand.rotation @ np.array([0.0, 0.0, 0.0])
    p_com_world = oM_hand.translation + oM_hand.rotation @ com_total_hand_local

    # Gravitaciona sila i torka u WORLD sistemu
    g_vec = np.array([0.0, 0.0, -9.81])
    F_grav = m_total * g_vec
    r_vec = p_com_world - p_sensor_world
    M_grav = np.cross(r_vec, F_grav)

    # Sirovi podaci sa senzora (u WORLD sistemu!)
    F_raw = mujoco_data.sensordata[force_addr:force_addr + 3]
    M_raw = mujoco_data.sensordata[torque_addr:torque_addr + 3]

    # Kompenzovana spoljna sila
    F_ext_clean = F_raw - F_grav
    M_ext_clean = M_raw - M_grav
    # Fext = np.concatenate([F_ext_clean, M_ext_clean])

    # Rotaciona matrica senzora
    R_sensor_to_world = oM_hand.rotation

    # Vektor od senzora do TCP u world koordinatama
    r_sensor_to_tcp = tcp_pos - p_sensor_world

    # koso simetricna matrica
    S_r_world = np.array([[0, -r_sensor_to_tcp[2], r_sensor_to_tcp[1]],
                          [r_sensor_to_tcp[2], 0, -r_sensor_to_tcp[0]],
                          [-r_sensor_to_tcp[1], r_sensor_to_tcp[0], 0]])

    # Transformaciona matrica prema formuli sa slike (za transformaciju u world sistem)
    T_sensor_to_world = np.zeros((6, 6))
    T_sensor_to_world[0:3, 0:3] = R_sensor_to_world  # R^w_s
    T_sensor_to_world[0:3, 3:6] = np.zeros((3, 3))  # O
    T_sensor_to_world[3:6, 0:3] = S_r_world @ R_sensor_to_world  # S(r^w_ws) R^w_s
    T_sensor_to_world[3:6, 3:6] = R_sensor_to_world  # R^w_s

    # Sila i moment na senzoru
    F_sensor_sensor = F_ext_clean
    M_sensor_sensor = M_ext_clean
    F_sensor_vec_sensor = np.concatenate([F_sensor_sensor, M_sensor_sensor])

    F_tcp_world = T_sensor_to_world @ F_sensor_vec_sensor

    F_ext_clean_world = F_tcp_world[0:3]
    M_ext_clean_world = F_tcp_world[3:6]

    Fext1 = np.concatenate([F_ext_clean_world, M_ext_clean_world])

    Fext_window.append(Fext1)
    if len(Fext_window) > window_size:
        Fext_window.pop(0)

    # filtriraj po komponentama
    Fext = np.mean(Fext_window, axis=0)  # ovo je 6D filtrirana sila i moment

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

    segment, t_seg = get_segment_index_and_local_time(t, segment_durations)

    if segment == 0:
        x_start_seg = home_tcp
        x_goal_seg = A
    else:
        idx = segment - 1
        x_start_seg = waypoints[idx]
        x_goal_seg = waypoints[idx + 1]

    #podaci za zeljenu poziciju,brzinu i ubrzanje iz Sinteze trajektorije
    x_d_pos, dx_d_pos, ddx_d_pos = sinteza_trajektorije(x_start_seg, x_goal_seg, t_seg, segment_durations[segment])
    x_d = np.concatenate([x_d_pos, rpy_const])
    dx_d = np.concatenate([dx_d_pos, np.zeros(3)])
    ddx_d = np.concatenate([ddx_d_pos, np.zeros(3)])
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

    tau = tau + tau_null  # ukupna torza
    # =============================== #

    mujoco_data.ctrl[:] = np.concatenate([tau[:7], [-200.0]])
    mujoco.mj_step(mujoco_model, mujoco_data)
    v.sync()

    # Snimanje podataka
    q_history.append(q.copy())
    dq_history.append(dq.copy())
    tau_history.append(tau.copy())
    tcp_pos_history.append(tcp_pos.copy())
    Fext_history.append(Fext.copy())
    time_history.append(t)
    k_history.append(k_squared)

# === Pretvaranje u numpy nizove === #
q_history = np.array(q_history)
dq_history = np.array(dq_history)
tau_history = np.array(tau_history)
tcp_pos_history = np.array(tcp_pos_history)
Fext_history = np.array(Fext_history)
time_history = np.array(time_history)
k_history = np.array(k_history)
Fext_raw_history = np.array(Fext_raw_history)
Fext_comp_history = np.array(Fext_comp_history)

plt.figure(figsize=(12,6))

# translacione sile
plt.plot(time_history, Fext_history[:,0], label='Fx')
plt.plot(time_history, Fext_history[:,1], label='Fy')
plt.plot(time_history, Fext_history[:,2], label='Fz')
plt.xlabel('Vreme [s]')
plt.ylabel('Sile [N]')
plt.legend()
plt.grid(True)
plt.show()

# momenti
plt.figure(figsize=(12,6))
plt.plot(time_history, Fext_history[:,3], label='Mx')
plt.plot(time_history, Fext_history[:,4], label='My')
plt.plot(time_history, Fext_history[:,5], label='Mz')

plt.xlabel('Vreme [s]')
plt.ylabel('Momenti [Nm]')
plt.legend()
plt.grid(True)
plt.show()


# 3D putanja TCP
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(tcp_pos_history[:,0], tcp_pos_history[:,1], tcp_pos_history[:,2], color='blue', label='Stvarna putanja')
ax.plot(ideal_path[:,0], ideal_path[:,1], ideal_path[:,2], color='green', linestyle='--', label='Idealna putanja')
waypoints_arr = np.array(waypoints)
ax.scatter(waypoints_arr[:,0], waypoints_arr[:,1], waypoints_arr[:,2], color='red', label='Tacke putanje')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()
ax.grid(True)
plt.show()

# Pozicije zglobova
plt.figure(figsize=(10,6))
for j in range(q_history.shape[1]):
    plt.plot(time_history, q_history[:,j], label=f'q {j+1}')
plt.xlabel('Vreme [s]')
plt.ylabel('Pozicije u zglobovima [rad]')
plt.legend()
plt.grid(True)
plt.show()