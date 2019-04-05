


from Models import Actor, Critic

import tensorflow as tf
from collections import deque
import random
import numpy as np

class ACAgent:
    def __init__(self,state_size, action_size, env, sess):
        self.env = env
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125

        self.memory = deque(maxlen=2000)
        #### ACTOR MODEL ####
        self.actor_state_input, self.actor = Actor(self.state_size, self.action_size).create_actor()
        _, self.target_actor = Actor(self.state_size, self.action_size).create_actor()

  
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size[0], 1, 1])
        actor_model_weights = self.actor.trainable_weights
        self.actor_grads = tf.gradients(self.actor.output, actor_model_weights, -tf.reshape(self.actor_critic_grad, shape=(1,3)))
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        #### CRITIC MODEL ####
        self.critic_state_input, self.critic_action_input, self.critic = Critic(self.state_size, self.action_size).create_critic()
        _,_, self.target_critic = Critic(self.state_size, self.action_size).create_critic()

        self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())
    
    def remember(self, cur_state, action, rewards, new_state, done):
        self.memory.append([cur_state, action, rewards, new_state, done])
    
    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, rewards, new_state, _ = sample
            cur_state = cur_state
            predicted_action = self.actor.predict(cur_state)[0].reshape(1,3,1,1)
            
            grads = self.sess.run(self.critic_grads, feed_dict = {
                self.critic_state_input : cur_state,
                self.critic_action_input : predicted_action
            })[0]
            
            self.sess.run(self.optimize, feed_dict = {
                self.actor_state_input : cur_state,
                self.actor_critic_grad : grads
            })
    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, rewards, new_state, done = sample
            
            if not done:
                
                target_action = self.target_actor.predict(new_state)[0].reshape(1,3,1,1)
                future_reward = self.target_critic.predict(
                    [new_state, target_action]
                )
                rewards += self.gamma*future_reward
                
            
            self.critic.fit([cur_state, action], rewards.reshape(1,1), verbose=0)
    def _train_critic_vector(self, samples, batch_size):
        cur_state = [t[0] for t in samples]
        action = [t[1] for t in samples]
        rewards = [t[2] for t in samples]
        new_state = [t[3] for t in samples]
        done = [t[4] for t in samples]

        target_action = self.target_actor.predict(new_state)[0].reshape(batch_size, 3, 1,1)
        future_reward = self.target_critic.predict(
            [new_state, target_action]
        )
        print(future_reward, future_reward.shape)
        print(rewards, rewards.shape)
    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        #self._train_critic_vector(samples, batch_size)
        self._train_actor(samples)
    def _update_actor_target(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau* actor_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)
    def _update_critic_target(self):
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau* critic_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.target_critic.set_weights(critic_target_weights)
    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()
    
    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            act = self.env.action_space.sample()
            return act/np.sum(act)
        return self.actor.predict(cur_state)[0]
        