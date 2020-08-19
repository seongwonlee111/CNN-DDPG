from environment.vrep_env import Env
from environment.env_modules import vrep
import time
import numpy as np
env = Env(19998)
# for i in range(1):
#     env.reset()
#     vrep.simxStopSimulation(env.clientID, vrep.simx_opmode_oneshot)
#     print('--------------------------------------')
# env.reset()
# vrep.simxPauseSimulation(env.clientID, vrep.simx_opmode_oneshot)

vrep.simxSynchronous(env.clientID, True)
time.sleep(0.5)
vrep.simxStartSimulation(env.clientID, vrep.simx_opmode_oneshot)
for i in range(5000):
    vrep.simxSynchronousTrigger(env.clientID)

    print(vrep.simxGetObjectPosition(env.clientID, env.target_handle ,env.ego_handle, vrep.simx_opmode_blocking)[1])