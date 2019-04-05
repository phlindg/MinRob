


from Models import Actor, Critic

import tensorflow as tf
from collections import deque
import random
import numpy as np

class Agent2:
    def __init__(self,state_size, action_size, env, sess):
        self.env = env
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.exploration = 1000
        self.gamma = 0.95
        self.tau = 0.125

        self.total_reward = 0
        self.loss = 0

        self.memory = deque(maxlen=2000)
        #### ACTOR MODEL ####
        self.actor_state_input, self.actor = Actor(self.state_size, self.action_size).create_actor()
        _, self.target_actor = Actor(self.state_size, self.action_size).create_actor()
        print(tf.shape(self.actor_state_input))
        print(self.actor_state_input.shape)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size[0], 1, 1])
        actor_model_weights = self.actor.trainable_weights
        
        self.actor_grads = tf.gradients(self.actor.output, actor_model_weights, -tf.reshape(self.actor_critic_grad, shape=(tf.shape(self.actor_state_input)[0],5)))
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        #### CRITIC MODEL ####
        self.critic_state_input, self.critic_action_input, self.critic = Critic(self.state_size, self.action_size).create_critic()
        _,_, self.target_critic = Critic(self.state_size, self.action_size).create_critic()

        self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())
    
    def remember(self, cur_state, action, rewards, new_state, done):
        self.memory.append([cur_state, action, rewards, new_state, done])
    
    def _train_actor(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict = {
            self.actor_state_input : states,
            self.actor_critic_grad : action_grads
        })
    def _train_actor_target(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)
    def critic_gradients(self, states, actions):
        return self.sess.run(self.critic_grads, feed_dict = {
            self.critic_state_input : states,
            self.critic_action_input : actions
        })[0]
    def _train_critic_target(self):
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.target_critic.set_weights(critic_target_weights)
    
    def act(self, cur_state):
        self.epsilon -= 1/self.exploration
        if np.random.random() < self.epsilon:
            act = np.random.rand(self.action_size[0])
            return act/np.sum(act)
        return self.actor.predict(cur_state)[0]
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.asarray([e[0] for e in minibatch]).reshape(batch_size, self.action_size[0], 50 ,1)
        actions = np.asarray([e[1] for e in minibatch]).reshape(batch_size, self.action_size[0], 1,1)
        rewards = np.asarray([e[2] for e in minibatch])
        new_states = np.asarray([e[3] for e in minibatch]).reshape(batch_size, 5, 50, 1)
        dones = np.asarray([e[4] for e in minibatch])


        future_action = self.target_actor.predict(new_states).reshape(batch_size, self.action_size[0], 1,1)
        target_q_values = self.target_critic.predict([new_states, future_action]).reshape(batch_size,)
        target = [0]*batch_size
        for k in range(batch_size):
            if dones[k]:
                target[k] = rewards[k]
            else:
                target[k] = rewards[k] + self.gamma*target_q_values[k]

        self.loss += self.critic.train_on_batch([states, actions], target)
        a_for_grad = self.actor.predict(states).reshape(batch_size,self.action_size[0],1,1)
        grads = self.critic_gradients(states, a_for_grad)
        self._train_actor(states, grads)
        self._train_actor_target()
        self._train_critic_target()