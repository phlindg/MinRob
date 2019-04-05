

from Agents import ACAgent
from Envs import PortfolioEnv
import tensorflow as tf
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

def main():
    sess = tf.Session()
    K.set_session(sess)
    sand = pd.read_csv("C:/Users/Phili/Desktop/fond/data/SAND.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
    eric = pd.read_csv("C:/Users/Phili/Desktop/fond/data/ERIC.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
    sand = sand["Closing price"]#.loc["2016-01-01":"2018-09-01"]
    eric = eric["Closing price"]#["2016-01-01":"2018-09-01"]
    volv = pd.read_csv("C:/Users/Phili/Desktop/fond/data/VOLV.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
    volv = volv["Closing price"]#["2016-01-01":"2018-09-01"]


    data = pd.DataFrame({"sand":sand, "eric":eric, "volv":volv}, index=sand.index)

    env = PortfolioEnv(data, steps=20, trading_cost=0.025, time_cost = 0.0, augment=0.1)
    state_size = env.observation_space.spaces["history"].shape
    action_size = env.action_space.shape
    action_size = (action_size[0],1,1)
    agent = ACAgent(state_size, action_size,env, sess)

    episode = 200
    for e in range(episode):
        weights = {s: [] for s in range(action_size[0])}
    
        cur_state = env.reset()
        action = env.action_space.sample()
        while True:
            cur_state = cur_state.reshape(1,3,50,1)
            action = agent.act(cur_state)
            action = action.reshape(1,3,1,1)
            new_state, reward, info, done = env.step(action)
            new_state = new_state.reshape(1,3,50,1)
            agent.remember(cur_state, action, reward, new_state, done)
            agent.train()
            agent.update_target()
            for s in range(action_size[0]):
                weights[s].append(action.reshape(3,)[s])

            cur_state = new_state
            if done:
                print("Episode: {}/{}, episode end value: {}, weights: {}".format(e+1, episode, info["portfolio_value"], action.reshape(3,)))
                break
        if e % 5 == 0:
            plt.subplot(1, episode/5, e/5+1)
            for s in range(action_size[0]):
                plt.plot(weights[s], label=str(s))
            plt.legend()


    
    plt.show()

main()
