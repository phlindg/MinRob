
from keras.layers import Dense, Conv2D, Input, concatenate, merge, Flatten, Activation
from keras.models import Model
from keras.optimizers import Adam

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
    def create_critic(self):
        price_input = Input(shape=self.state_size)
        feature_maps = Conv2D(2, (1,3), activation="relu")(price_input)
        feature_maps = Conv2D(20, (1,48), activation="relu")(feature_maps)

        w_last = Input(shape=self.action_size)
        feature_maps = concatenate([feature_maps, w_last], axis=-1)
        feature_map = Conv2D(1, (1,1), activation="relu")(feature_maps)
        feature_map = Flatten()(feature_map)
        output = Dense(1, activation="relu")(feature_map)


        model = Model(inputs = [price_input, w_last], outputs=output)
        adam = Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())
        self.model = model
        return price_input,w_last, model
    def create_critic_lstm(self):
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
        return price_input, w_last, model







