import numpy as np
import tensorflow as tf
import math

TAU = 0.001
LEARNING_RATE= 0.001
BATCH_SIZE = 64
class CriticNet:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self,num_states,num_actions):
        
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #critic_q_model parameters:
            self.Conv1_c, self.Conv2_c, self.W1_c, self.W1p_c, self.B1_c, self.W1_action_c, self.W2_c, self.B2_c,\
            self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(num_states, num_actions)
                                   
            #create target_q_model:
            self.t_Conv1_c, self.t_Conv2_c, self.t_W1_c, self.t_W1p_c, self.t_B1_c, self.t_W1_action_c, self.t_W2_c, self.t_B2_c,\
            self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(num_states, num_actions)
            
            self.q_value_in=tf.placeholder("float",[None,1]) #supervisor
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c) 
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2_c,2))             
            self.cost=tf.pow(self.critic_q_model-self.q_value_in,2)/BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            
            #action gradient to be used in actor network:
            #self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            #from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])] #this is just divided by batch size
            #from simple actor net:
            self.check_fl = self.action_gradients             
                       
            #initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())
            
            #To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_Conv1_c.assign(self.Conv1_c),
                self.t_Conv2_c.assign(self.Conv2_c),
				self.t_W1_c.assign(self.W1_c),
                self.t_W1p_c.assign(self.W1p_c),
				self.t_B1_c.assign(self.B1_c),
                self.t_W1_action_c.assign(self.W1_action_c),
				self.t_W2_c.assign(self.W2_c),
				self.t_B2_c.assign(self.B2_c),
			])
            
            self.update_target_critic_op = [
                self.t_Conv1_c.assign(TAU*self.Conv1_c+(1-TAU)*self.t_Conv1_c),
                self.t_Conv2_c.assign(TAU*self.Conv2_c+(1-TAU)*self.t_Conv2_c),
                self.t_W1_c.assign(TAU*self.W1_c+(1-TAU)*self.t_W1_c),
                self.t_W1p_c.assign(TAU*self.W1p_c+(1-TAU)*self.t_W1p_c),
                self.t_B1_c.assign(TAU*self.B1_c+(1-TAU)*self.t_B1_c),
                self.t_W1_action_c.assign(TAU*self.W1_action_c+(1-TAU)*self.t_W1_action_c),
                self.t_W2_c.assign(TAU*self.W2_c+(1-TAU)*self.t_W2_c),
                self.t_B2_c.assign(TAU*self.B2_c+(1-TAU)*self.t_B2_c)
            ]
            self.saver = tf.train.Saver()

            
    def create_critic_net(self, num_states=4, num_actions=2):
        N_HIDDEN_1 = 200
        N_HIDDEN_2 = 200
        critic_state_in = tf.placeholder("float",[None,96*4+2])
        critic_action_in = tf.placeholder("float",[None,num_actions])    
    
        Conv1_c = tf.Variable(tf.random_uniform([8, 4, 16], -0.0003, 0.0003))
        Conv2_c = tf.Variable(tf.random_uniform([4, 16, 32], -0.0003, 0.0003))
        W1_c = tf.Variable(tf.random_uniform([12*32,N_HIDDEN_1],-1/math.sqrt(12*32),1/math.sqrt(12*32)))
        W1p_c = tf.Variable(tf.random_uniform([2,N_HIDDEN_1],-1/math.sqrt(2),1/math.sqrt(2)))
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(12*32+num_actions),1/math.sqrt(12*32+num_actions)))   
        W1_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_1],-1/math.sqrt(num_actions),1/math.sqrt(num_actions)))
        W2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
        B2_c= tf.Variable(tf.random_uniform([1],-0.003,0.003))

        lidar_state_in = critic_state_in[:,:96*4]
        lidar_state_in = tf.reshape(lidar_state_in, [tf.shape(lidar_state_in)[0], 96, 4])
        prev_action_state_in = critic_state_in[:,96*4:]
    
        C1_c = tf.nn.conv1d(lidar_state_in, Conv1_c, 4, padding='SAME')
        C1_c = tf.nn.softplus(C1_c)
        C2_c = tf.nn.conv1d(C1_c, Conv2_c, 2, padding='SAME')
        C2_c = tf.nn.softplus(C2_c)
        C2_c = tf.layers.flatten(C2_c)
        H1_c = tf.nn.softplus(tf.matmul(C2_c,W1_c)+tf.matmul(prev_action_state_in,W1p_c) +tf.matmul(critic_action_in,W1_action_c)+B1_c)
        critic_q_model=tf.matmul(H1_c,W2_c)+B2_c

        return Conv1_c, Conv2_c, W1_c, W1p_c, B1_c, W1_action_c, W2_c, B2_c, critic_q_model, critic_state_in, critic_action_in
    
    def train_critic(self, state_t_batch, action_batch, y_i_batch ):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in:action_batch, self.q_value_in: y_i_batch})
             
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})    
        
    def compute_delQ_a(self,state_t,action_t):
#        print '\n'
#        print 'check grad number'        
#        ch= self.sess.run(self.check_fl, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})
#        print len(ch)
#        print len(ch[0])        
#        raw_input("Press Enter to continue...")        
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)

    def save_critic(self, save_path):
        save_path = self.saver.save(self.sess, save_path)
    def load_critic(self, save_path):
        self.saver.restore(self.sess, save_path)
