


import numpy as np
import math
#from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network()  
        self.target_model, self.target_action, self.target_state = self.create_critic_network()  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self):
        price_input = Input(shape=self.state_size)
        x = Dense(units=100, activation="relu")(price_input)
        x = Dense(units=1, activation="relu")(x)
        
        w_last = Input(shape=self.action_size)
        x_w = Dense(units=1, activation="relu")(w_last)

        conc = concatenate([x, x_w], axis=2)
        x = Dense(units=25, activation="relu")(conc)
        x = Flatten()(x)
        output = Dense(units=1, activation="relu")(x)

        model = Model(inputs=[price_input, w_last], outputs = output)
        adam = Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())
        return model, w_last, price_input