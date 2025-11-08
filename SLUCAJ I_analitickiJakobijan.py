import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer
from robot_descriptions.loaders.pinocchio import load_robot_description as load_pinocchio
from robot_descriptions.loaders.mujoco import load_robot_description as load_mujoco
from Funkcije import sinteza_trajektorije, matrica_T, izvod_matrice_T, matrica_T_inv,izvod_matriceT_inv

# ===Ucitavanje modela=== #
# ---pinocchio model--- #
pinocchio_robot = load_pinocchio("panda_description")
pinocchio_model = pinocchio_robot.model
pinocchio_data = pinocchio_robot.data
# ---mujoco model--- #
mujoco_model = load_mujoco("panda_mj_description")
mujoco_data = mujoco.MjData(mujoco_model)

# ===Podesavanje simulacije=== #
dt = 0.001
T = 10.0
T_traj = 3
N = int(T/dt)
mujoco_model.opt.timestep = dt
# ---pocetne konfiguracije--- #
nq = pinocchio_model.nq
ndq = pinocchio_model.nv
q0 = np.zeros(nq)
dq0 = np.zeros(ndq)
mujoco_data.qpos[:] = q0.copy()
mujoco_data.qvel[:] = dq0.copy()
v = viewer.launch_passive(mujoco_model,mujoco_data)

# ===TCP alata(hvataljke) u urdf fajlu - koordinatni sistem na kraju effektora=== #
frame = "panda_hand_tcp"
frame_id = pinocchio_model.getFrameId(frame)

# ===Parametri impedansnog upravljanja=== #
# Hm = np.diag([1000, 1000, 1000, 500, 500, 500])
# Dm = np.diag([2000, 2000, 2000, 1000, 1000, 1000])
# Km = np.diag([5000, 5000, 5000, 3000, 3000, 3000])

# === Parametri impedanse preko prirodne ucestanosti i prigusenja === #
# ---Zeljena krutost--- #
K_trans = np.array([400, 400, 300])  # translacija
K_rot   = np.array([300, 300, 300])  # rotacija
K_diag  = np.concatenate([K_trans, K_rot])
Km = np.diag(K_diag)

# ---Prirodna ucestanost--- #
wn_trans = np.array([5.0, 5.0, 4.0])   # translacione ose
wn_rot   = np.array([4.0, 4.0, 4.0])   # rotacione ose
wn_diag  = np.concatenate([wn_trans, wn_rot])

# ---Zeljena inercija--- #
Hm_diag = K_diag / (wn_diag**2)
Hm = np.diag(Hm_diag)

# ---Prigusenje--- #
zeta = 1.0
Dm_diag = 2.0 * zeta * (K_diag / wn_diag)
Dm = np.diag(Dm_diag)

# ---Parametri trajektorije, pocetni i krajnji target trajektorije--- #
# x_start = np.array([0.72, -0.2, 0.3,0.0,0.0,np.pi])
# x_goal = np.array([0.72, 0.2, 0.3,0.0,0.0,np.pi])
x_d = np.array([0.6, -0.2, 0.3]) #pozicija targeta
r_d = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0) #orijentacija targeta
dx_d = np.zeros(6)
ddx_d = np.zeros(6)
Fext = np.zeros(6)

# === Merenje kontaktne sile === #
#---Pronadji indekse senzora po imenu--- #
# force_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "finger_tip_force")
# torque_sensor_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "finger_tip_torque")
#
# # ---Dobavi adrese u sensordata nizu--- #
# force_addr = mujoco_model.sensor_adr[force_sensor_id]
# torque_addr = mujoco_model.sensor_adr[torque_sensor_id]

for i in range(N):

    t = i * dt #vreme simulacije

    #Citam stanje iz Mujoca
    q = mujoco_data.qpos[:].copy()
    dq = mujoco_data.qvel[:].copy()
    mujoco.mj_forward(mujoco_model,mujoco_data)

    # Podaci o end effectoru iz Pinocchia
    pin.framesForwardKinematics(pinocchio_model, pinocchio_data, q)
    poz_orij_hvataljke = pinocchio_data.oMf[frame_id]
    pozicija_hvataljke = poz_orij_hvataljke.translation.copy()
    orijentacija_hvataljke = poz_orij_hvataljke.rotation.copy()

    #Racunanje geometrijskog
    J = pin.computeFrameJacobian(pinocchio_model, pinocchio_data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    ####### ANALITICKI JAKOBIJAN ######
    psi, teta, fi = pin.rpy.matrixToRpy(orijentacija_hvataljke)
    T_inv = matrica_T_inv(psi,teta,fi)
    I = np.eye(3)
    T_a = np.block([
        [I, np.zeros((3, 3))],
        [np.zeros((3, 3)), T_inv]
    ])
    J_a = T_a @ J
    ####### ANALITICKI JAKOBIJAN ######

    ####### IZVOD ANALITICKOG JAKOBIJANA ######
    dJ = pin.frameJacobianTimeVariation(pinocchio_model, pinocchio_data, q, dq, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    dx = J_a @ dq
    dpsi = dx[3]
    dteta = dx[4]
    dfi = dx[5]
    dT = izvod_matriceT_inv(psi, teta, fi, dpsi, dteta, dfi)
    dT_a = np.block([
        [np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), dT]
    ])
    dJ_a = T_a @ dJ + dT_a @ J
    ####### IZVOD ANALITICKOG JAKOBIJANA ######

    ###### PSEUDOINV. ANALIT. JAKOBIJAN ######
    U, s, Vt = np.linalg.svd(J_a)
    sigma_min = np.min(s)
    epsilon = 0.03
    k_max = 0.3
    if sigma_min >= epsilon:
        k_squared = 0.0
    else:
        k_squared = (1 - (sigma_min / epsilon) ** 2) * k_max ** 2

    J_a_inv = J_a.T @ np.linalg.inv(J_a @ J_a.T + k_squared * np.eye(6))
    ###### PSEUDOINV. ANALIT. JAKOBIJAN ######

    # F = mujoco_data.sensordata[force_addr:force_addr + 3]
    # M = mujoco_data.sensordata[torque_addr:torque_addr + 3]
    # Fext = np.concatenate([F, M])
    Fext = np.zeros(6)
    F_a = T_a.T @ Fext

    #Sinteza_trajektorije vraća željenu poziciju, brzinu i ubrzanje u trenutku t
    # r_d = pin.rpy.rpyToMatrix(x_start[3:]).copy()

    #Racunanje greske pozicije i orijentacije
    x_error = np.concatenate([x_d[:3] - pozicija_hvataljke, pin.rpy.matrixToRpy(r_d@orijentacija_hvataljke.T)])
    #x_error = np.concatenate([x_d[:3] - pozicija_hvataljke, pin.log3(r_d @ orijentacija_hvataljke.T)])
    dx_error = dx_d - dx

    #Potrebni izrazi za implementaciju upravljanja
    H = pinocchio_robot.mass(q)
    h = pinocchio_robot.nle(q, dq)
    g = pinocchio_robot.gravity(q)
    h_bez_gravitacije = h - g

    Hx = np.linalg.inv(J_a @ np.linalg.inv(H) @ J_a.T)
    hx = Hx@ J_a @ np.linalg.inv(H) @ h_bez_gravitacije - Hx @ dJ_a @ dq
    gx = Hx @ J_a @ np.linalg.inv(H) @ g
    a = ddx_d + np.linalg.inv(Hm) @ (Dm @ (dx_error)+Km@(x_error)+F_a)

    #upravljanje
    tau_ruke = J_a.T @(Hx@a+hx+gx+F_a)
    tau_hvataljke = np.array([0.0])
    tau = np.concatenate([tau_ruke[:7], tau_hvataljke])
    mujoco_data.ctrl[:] = tau
    mujoco.mj_step(mujoco_model, mujoco_data)
    v.sync()

print("Konacna pozicija hvataljke je:", pozicija_hvataljke)
print("Konacna orijentacija hvataljke je:", pin.rpy.matrixToRpy(orijentacija_hvataljke))