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
T = 10.0
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
K_trans = np.array([10000, 10000, 10000])
K_rot   = np.array([100, 100, 100])
K_diag  = np.concatenate([K_trans, K_rot])
Km = np.diag(K_diag)
wn_trans = np.array([10., 10., 1.])
wn_rot   = np.array([16., 16., 16.])
wn_diag  = np.concatenate([wn_trans, wn_rot])
Hm_diag = K_diag / (wn_diag**2)
print(Hm_diag)
Hm = np.diag(Hm_diag)
zeta = 1.0
Dm_diag = 2.0 * zeta * (K_diag / wn_diag)
Dm = np.diag(Dm_diag)

# === Pravougaonik za crtanje === #
z_const = 0.24
rpy_const = np.array([np.pi, 0.0, 0.0])
A = np.array([0.62, -0.2, z_const])
B = np.array([0.72, -0.2, z_const])
C = np.array([0.72, 0.2, z_const])
D = np.array([0.62, 0.2, z_const])
waypoints = [A, B, C, D, A]
num_segments = len(waypoints)-1
T_traj_total = 10.0
T_side = T_traj_total / num_segments

# === Liste za prikupljanje podataka === #
q_history, dq_history, tau_history = [], [], []
tcp_pos_history, Fext_history, time_history = [], [], []

# === Generisanje idealne putanje === #
num_points_per_segment = 100
ideal_path = []

for i in range(num_segments):
    x_start_seg = waypoints[i]
    x_goal_seg = waypoints[i+1]
    for t_step in np.linspace(0, T_side, num_points_per_segment):
        x_d_pos, _, _ = sinteza_trajektorije(x_start_seg, x_goal_seg, t_step, T_side)
        ideal_path.append(x_d_pos)

ideal_path = np.array(ideal_path)

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

    #Kontaktna sila
    F = mujoco_data.sensordata[force_addr:force_addr + 3]
    M = mujoco_data.sensordata[torque_addr:torque_addr + 3]
    Fext = np.concatenate([F, M])
    print(Fext)

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

    segment = int(t // T_side)
    if segment >= num_segments:
        segment = num_segments-1
    t_seg = t - segment * T_side
    t_seg = min(t_seg, T_side)
    x_start_seg = waypoints[segment]
    x_goal_seg = waypoints[segment+1]

    #podaci za zeljenu poziciju,brzinu i ubrzanje iz Sinteze trajektorije
    x_d_pos, dx_d_pos, ddx_d_pos = sinteza_trajektorije(x_start_seg, x_goal_seg, t_seg, T_side)
    x_d = np.concatenate([x_d_pos, rpy_const])
    dx_d = np.concatenate([dx_d_pos, np.zeros(3)])
    ddx_d = np.concatenate([ddx_d_pos, np.zeros(3)])

    #
    r_d = pin.rpy.rpyToMatrix(rpy_const)
    # x_error = np.concatenate([x_d[:3]-tcp_pos, pin.log3(r_d @ tcp_rot.T)])
    # --- pozicioni deo (isti) ---
    pos_error = x_d[:3] - tcp_pos

    # --- ugaona greška u lokalnom okviru end-effectora ---
    rot_err_local = pin.log3(tcp_rot.T@r_d)  # vraća vektor u end-effector (lokal) okviru

    # --- prebaci ugaonu grešku u bazu/world koristeći tcp_rot (R_world_end) ---
    rot_err_world = tcp_rot @ rot_err_local

    # --- kompletan error u world okviru ---
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

# === Plot rezultata === #
# Kontaktne sile
plt.figure(figsize=(10,6))
# plt.plot(time_history, Fext_history[:,0], label='Fx')
# plt.plot(time_history, Fext_history[:,1], label='Fy')
plt.plot(time_history, Fext_history[:,2], label='Fz')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Kontaktne sile na TCP')
plt.legend()
plt.grid(True)
plt.show()

# 3D putanja TCP
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(tcp_pos_history[:,0], tcp_pos_history[:,1], tcp_pos_history[:,2], color='blue', label='TCP stvarna putanja')
ax.plot(ideal_path[:,0], ideal_path[:,1], ideal_path[:,2], color='green', linestyle='--', label='Idealna putanja')
waypoints_arr = np.array(waypoints)
ax.scatter(waypoints_arr[:,0], waypoints_arr[:,1], waypoints_arr[:,2], color='red', label='Waypoints')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D putanja TCP')
ax.legend()
ax.grid(True)
plt.show()

# Pozicije zglobova
plt.figure(figsize=(10,6))
for j in range(q_history.shape[1]):
    plt.plot(time_history, q_history[:,j], label=f'Joint {j+1}')
plt.xlabel('Time [s]')
plt.ylabel('Joint angles [rad]')
plt.title('Pozicije zglobova')
plt.legend()
plt.grid(True)
plt.show()
