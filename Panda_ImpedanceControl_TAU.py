import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer
from robot_descriptions.loaders.pinocchio import load_robot_description as load_pinocchio
from robot_descriptions.loaders.mujoco import load_robot_description as load_mujoco
from Funkcije import sinteza_trajektorije

#Ucitavanje modela
#pinocchio model
pinocchio_robot = load_pinocchio("panda_description")
pinocchio_model = pinocchio_robot.model
pinocchio_data = pinocchio_robot.data
#mujoco model
mujoco_model = load_mujoco("panda_mj_description_box")
mujoco_data = mujoco.MjData(mujoco_model)


#Podesavanje simulacije
dt = 0.001
T = 10.0
T_traj = 3
N = int(T/dt)
mujoco_model.opt.timestep = dt
#pocetne konfiguracije
nq = pinocchio_model.nq
ndq = pinocchio_model.nv
q0 = np.zeros(nq)
dq0 = np.zeros(ndq)
mujoco_data.qpos[:] = q0.copy()
mujoco_data.qvel[:] = dq0.copy()
v = viewer.launch_passive(mujoco_model,mujoco_data)

#TCP alata(hvataljke) u urdf fajlu - koordinatni sistem na kraju effektora
frame = "panda_hand_tcp"
frame_id = pinocchio_model.getFrameId(frame)

#Parametri impedansnog upravljanja
Hm = np.diag([100, 100, 100, 100, 100, 100])
Dm = np.diag([2500, 2500, 2500, 1200, 1200, 1200])
Km = np.diag([10000, 10000, 10000, 7000, 7000, 7000])

#Parametri trajektorije, pocetni i krajnji target trajektorije
x_start = np.array([0.7, -0.2, 0.37,np.pi,0.0,0.0]) #dodao sam kutiju koja je visine 0.4
#kutiju sam dodao u xml fajl i  to izgleda ovako
#<geom name="large_box"
        #  type="box"
       #   size="0.2 0.4 0.2"
       #   pos="0.7 0.0 0.2"
        #  rgba="0.8 0.3 0.3 0.7"/>
x_goal = np.array([0.7, 0.2, 0.38,np.pi,0.0,0.0])

#Test konverzije rpy uglova, provera konzistentnosti pinocchio biblioteke
#cilj je proveriti da li direktna i inverzna transformacija rpy daju iste rezultate
test_rpy = x_start[3:] # uzimanje RPY komponenti iz početne tačke
test_matrix = pin.rpy.rpyToMatrix(test_rpy) # Konverzija RPY uglova u rotacionu matricu
back_to_rpy = pin.rpy.matrixToRpy(test_matrix) # Obrnuta konverzija u RPY uglove
print("Test RPY konverzije:")
print(f"Original RPY: {test_rpy}")
print(f"Matrix:\n{test_matrix}")
print(f"Vraćeno RPY: {back_to_rpy}")
print(f"Razlika: {test_rpy - back_to_rpy}")

#inicijalizacija spoljasnje sile, probao sam da testiram sa silom kada je nula i kada je ukljucim dole u kodu
#Fext = np.zeros(6)

for i in range(N):
    t = i * dt #vreme simulacije

    #Citam stanje iz Mujoca
    q = mujoco_data.qpos[:].copy()
    dq = mujoco_data.qvel[:].copy()
    mujoco.mj_forward(mujoco_model,mujoco_data)

    #Merenje spoljasnjih sila tj kontakata sa okolinom pomocu mujoco
    # Ova metoda prebrojava sve kontakte i računa ukupne sile i momente
    contact_forces = []
    for id, contact in enumerate(mujoco_data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(mujoco_model, mujoco_data, id, force)
        contact_forces.append(force.copy())
    total_contact_force = np.sum([f[:3] for f in contact_forces], axis=0) if contact_forces else np.zeros(3)
    total_contact_torque = np.sum([f[3:] for f in contact_forces], axis=0) if contact_forces else np.zeros(3)
    force_magnitude = np.linalg.norm(total_contact_force)
    Fext = np.concatenate([total_contact_force, total_contact_torque])

    #Podaci o end effectoru
    pin.framesForwardKinematics(pinocchio_model, pinocchio_data, q) # Računanje kinematike svih frame-ova
    poz_orij_hvataljke = pinocchio_data.oMf[frame_id]
    pozicija_hvataljke = poz_orij_hvataljke.translation.copy()
    orijentacija_hvataljke = poz_orij_hvataljke.rotation.copy()

    #Jakobijan
    J = pin.computeFrameJacobian(pinocchio_model, pinocchio_data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    dJ = pin.frameJacobianTimeVariation(pinocchio_model, pinocchio_data, q, dq, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    pInvJ = J.T @ np.linalg.inv(J @ J.T + 0.3**2 * np.eye(6))

    #Brzina vrha hvataljke
    dx = J @ dq

    #Sinteza_trajektorije vraća željenu poziciju, brzinu i ubrzanje u trenutku t
    x_d, dx_d, ddx_d = sinteza_trajektorije(x_start, x_goal, t, T_traj)
    # dx_d = np.concatenate([dx_d[:3], np.zeros(3)])
    # ddx_d = np.concatenate([ddx_d[:3], np.zeros(3)])

    #Racunanje greske pozicije i orijentacije
    r_d = pin.rpy.rpyToMatrix(x_d[3:])
    #x_error = np.concatenate([x_d[:3] - pozicija_hvataljke, pin.rpy.matrixToRpy(r_d@orijentacija_hvataljke.T)])
    x_error = np.concatenate([x_d[:3] - pozicija_hvataljke, pin.log3(r_d @ orijentacija_hvataljke.T)])
    dx_error = dx_d - dx

    ###
    #Jednacina za tau jer je konacna implementacija upravljanja
    #uvek u prostoru zgloba robota, rastavicu deo po deo, s tim da mi deo h sadrzi i dq i g
    ###

    H = pinocchio_robot.mass(q)
    h = pinocchio_robot.nle(q, dq)
    deo11 = H @ pInvJ
    deo12 = ddx_d - dJ @ dq + np.linalg.inv(Hm) @ (Dm @ (dx_error) + Km @ (x_error))
    deo1 = deo11 @ deo12
    deo2 = h
    deo41 = H@pInvJ@np.linalg.inv(Hm) - J.T
    deo42 = Fext
    deo4 = deo41 @ Fext

    tau_ruke = deo1 + deo2  + deo4
    tau_hvataljke = np.array([0.0])
    tau = np.concatenate([tau_ruke[:7],tau_hvataljke])
    mujoco_data.ctrl[:] = tau
    mujoco.mj_step(mujoco_model, mujoco_data)
    v.sync()

print("Konacna pozicija hvataljke je:", pozicija_hvataljke)
print("Konacna orijentacija hvataljke je:", pin.rpy.matrixToRpy(orijentacija_hvataljke))


