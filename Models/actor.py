
from keras.layers import Dense, Conv2D, Input, concatenate, merge, Flatten, Activation, LSTM
from keras.models import Model
from keras.optimizers import Adam

class Actor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
    def create_actor(self):
        price_input = Input(shape=self.state_size)
        feature_maps = Conv2D(2, (1,3), activation="relu")(price_input)
        feature_maps = Conv2D(20, (1,48), activation="relu")(feature_maps)

        feature_map = Conv2D(1, (1,1), activation="relu")(feature_maps)
        feature_map = Flatten()(feature_map)
        w = Activation("tanh")(feature_map)

        model = Model(inputs = price_input, outputs=w)
        adam = Adam(lr = 1e-3)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())
        self.model = model
        return price_input, model
    def create_actor_lstm(self):
        price_input = Input(shape=self.state_size)
        lstm1 = LSTM(20)(price_input)
        w = Activation("softmax")(lstm1)

        model = Model(inputs = price_input, outputs = w)
        adam = Adam(lr = 1e-3)
        model.compile(loss="mse", optimizer = adam)
        print(model.summary)
        self.model = model
        return price_input, model
    
