
import numpy as np
import math
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Conv2D, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network()   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network()

        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size[0], 1, 1])
        
        self.params_grad = tf.gradients(self.model.output, self.weights, -tf.reshape(self.action_gradient, shape=(1,3)))
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)

        
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self):
        price_input = Input(shape=self.state_size)
        feature_maps = Conv2D(2, (1,3), activation="relu")(price_input)
        feature_maps = Conv2D(20, (1,48), activation="relu")(feature_maps)

        feature_map = Conv2D(1, (1,1), activation="relu")(feature_maps)
        feature_map = Flatten()(feature_map)
        w = Activation("softmax")(feature_map)

        model = Model(inputs = price_input, outputs=w)
        adam = Adam(lr = 1e-3)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())
        self.model = model
        return model, model.trainable_weights, price_input
    
        

