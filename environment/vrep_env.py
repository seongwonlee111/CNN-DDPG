from .env_modules import vrep
import random
import numpy as np
import time

class Env():
    def __init__(self, port):
        self.clientID = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)
       
        # Set handles
        self.drone_handle = vrep.simxGetObjectHandle(self.clientID, 'Quadricopter', vrep.simx_opmode_blocking)[1]
        self.drone_target_handle = vrep.simxGetObjectHandle(self.clientID, 'Quadricopter_target', vrep.simx_opmode_blocking)[1]
        self.yaw_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'yaw', vrep.simx_opmode_blocking)[1]
        self.pitch_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'pitch', vrep.simx_opmode_blocking)[1]
        self.vision_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor', vrep.simx_opmode_blocking)[1]
        self.collision_handles = vrep.simxGetCollisionHandle(self.clientID, 'Collision', vrep.simx_opmode_blocking)[1]

        self.vision_sensor_pose_init = {'yaw':[],'pitch':[]}
        self.drone_pose_init = [-0.85, -0.15, 0.3]

        self.state_vision_image = []
        self.state_pitch_velocity = 0.
        self.state_yaw_velocity = 0.

        self.stepsize = 0.01
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxGetObjectPosition(self.clientID, self.drone_handle, -1. vrep.simx_opmode_streaming)
        vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_streaming)

    def get_image(self):
        vision_array = []
        vision_array = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_buffer)[2]

        return vision_array
    
    def computeXYZ(self,drone_position):
        distance = np.sqrt(drone_position[0]**2+drone_position[1]**2)
        x = random.uniform(-self.stepsize,self.stepsize))
        y = random.uniform(-self.stepsize,self.stepsize))
        z = random.uniform(-self.stepsize,self.stepsize))

        if drone_position[0] < - 1.5 :
            x = random.uniform(self.stepsize/3.,self.stepsize))
        elif drone_position[0] > 1.5 :
            x = random.uniform(-self.stepsize,-self.stepsize/3.))
        else:
            x = random.uniform(-self.stepsize,self.stepsize))

        if drone_position[1] < - 1.5 :
            x = random.uniform(0.,self.stepsize))
        elif drone_position[1] > 1.5 :
            x = random.uniform(-self.stepsize,-self.stepsize/3.))
        else:
            x = random.uniform(-self.stepsize,self.stepsize))

        if drone_position[2] < 0.28 :
            x = random.uniform(self.stepsize/3.,self.stepsize))
        elif drone_position[2] > 0.45 :
            x = random.uniform(-self.stepsize,-self.stepsize/3.))
        else:
            x = random.uniform(-self.stepsize,self.stepsize))
       

        return x, y, z

    def reset(self):
        vrep.simxSynchronous(self.clientID, True)
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        vrep.simxSetObjectPosition(self.clientID, self.drone_target_handle, -1, self.drone_pose_init, vrep.simx_opmode_oneshot)  
        vrep.simxSetObjectPosition(self.clientID, self.drone_handle, -1, self.drone_pose_init, vrep.simx_opmode_oneshot)  
        vrep.simxSetObjectOrientation(self.clientID, self.yaw_joint_handle, -1, vision_sensor_pose_init['yaw'], vrep.simx_opmode_oneshot)        
        vrep.simxSetObjectOrientation(self.clientID, self.pitch_joint_handle, -1, vision_sensor_pose_init['pitch'], vrep.simx_opmode_oneshot)       

    def get_reward(self,image):
        reward = 0.
        
        return reward 

    def step(self, action, timestep):
        done = False
        drone_position = np.array(vrep.simxGetObjectPosition(self.clientID, self.drone_handle, -1. vrep.simx_opmode_buffer)[1])
        x, y, z = self.computeXYZ(drone_position)
        vrep.simxSetObjectPosition(self.clientID, self.drone_target_handle, -1, drone_position + np.array([]), vrep.simx_opmode_oneshot)

        vrep.simxPauseCommunication(self.clientID, 1)
        self.set_action(action)
        self.move_drone(drone_position)
        vrep.simxPauseCommunication(self.clientID, 0)

        vrep.simxSynchronousTrigger(self.clientID)

        self.state_vision_image = get_image()
        self.state_yaw_velocity = vrep.simxGetObjectVelocity(self.clientID, self.yaw_joint_handle, vrep.simx_opmode_blocking)[1]
        self.state_pitch_velocity = vrep.simxGetObjectVelocity(self.clientID, self.pitch_joint_handle, vrep.simx_opmode_blocking)[1]

        state = self.state_vision_image

        # Get collision 
        collision = {}
        collision = vrep.simxReadCollision(self.clientID, self.collision_handles, vrep.simx_opmode_blocking)[1]
        
        reward = get_reward(state_vision_image)


        if collision:
            vrep.simxSynchronousTrigger(self.clientID)
            done = True

        if done == True:
            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action([0.0, 0.0])
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

        self.prev_state_vision_image = self.state_vision_image
        self.prev_state_yaw_velocity = self.state_yaw_velocity
        self.prev_state_pitch_velocity = self.state_pitch_velocity

        return state, reward, done

    def set_action(self, action):
        yaw_speed = action[0]
        pitch_speed = action[1]
        vrep.simxSetJointTargetVelocity(self.clientID, self.yaw_joint_handle, yaw_speed, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(self.clientID, self.pitch_joint_handle, pitch_speed, vrep.simx_opmode_oneshot)

    def close(self):
        for name in self.collision_list:
            vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_discontinue)
        vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_discontinue)
        for name in self.lidar_list:
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(0.5)        