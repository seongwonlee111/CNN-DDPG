import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from environment.vrep_env import Env

episodes=30000
is_batch_norm = False #batch normalization switch

def main():
    env= Env(19997)
    steps = 300
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(2)
    counter = 0
    reward_per_episode = 0.
    num_states = 32*16
    num_actions = 2

    #saving reward:
    reward_st = np.array([0])
    
    for i in range(episodes):
        print ("==== Starting episode no:",str(i),"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise
            
            observation,reward,done=env.step(action,t)
            agent.add_experience(x,observation,action,reward,done)

            if counter > 64:
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done):
                print ('EPISODE: ',str(i),' Steps: ',str(t),' Total Reward: ',str(reward_per_episode))
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                agent.actor_net.save_actor('/home/lee/Projects/Tracking/RL/weights/actor/model.ckpt')
                agent.critic_net.save_critic('/home/lee/Projects/Tracking/RL/weights/critic/model.ckpt')
                print ('\n\n')
                break

if __name__ == '__main__':
    main()    