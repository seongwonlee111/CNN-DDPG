import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from environment.vrep_env import Env
episodes=200
is_batch_norm = False

def main():
    env= Env(20000)
    steps = 50    
    agent = DDPG(env, is_batch_norm)
    counter=0
    exploration_noise = OUNoise(2)
    reward_per_episode = 0    
    num_states = 96*4+4
    num_actions = 2
    reward_st = np.array([0])

    agent.actor_net.load_actor('/home/myounghoe/ddpgtf/norepeat_target_2action_scale2/weights/actor/model.ckpt')
    agent.critic_net.load_critic('/home/myounghoe/ddpgtf/norepeat_target_2action_scale2/weights/critic/model.ckpt')

    for i in range(episodes):
        print ("==== Starting episode no:",str(i),"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0]# + noise
            action = np.array([-1.0, 0.0])
            observation,reward,done=env.step(action,t)
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done):
                print ('EPISODE: ',str(i),' Steps: ',str(t),' Total Reward: ',str(reward_per_episode))
                # print "Printing reward to file"
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('test_reward.txt',reward_st, newline="\n")
                print ('\n\n')
                break

if __name__ == '__main__':
    main()    