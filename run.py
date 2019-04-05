


from Models import ActorNetwork, CriticNetwork
from Agents import Agent2
from Envs import PortfolioEnv

import pandas as pd
import numpy as np
import tensorflow as tf

sand = pd.read_csv("C:/Users/Phili/Desktop/fond/data/SAND.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
eric = pd.read_csv("C:/Users/Phili/Desktop/fond/data/ERIC.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
sand = sand["Closing price"]#.loc["2016-01-01":"2018-09-01"]
eric = eric["Closing price"]#["2016-01-01":"2018-09-01"]
volv = pd.read_csv("C:/Users/Phili/Desktop/fond/data/VOLV.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
volv = volv["Closing price"]#["2016-01-01":"2018-09-01"]
hm = pd.read_csv("C:/Users/Phili/Desktop/fond/data/HM.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
hm = hm["Closing price"]#["2016-01-01":"2018-09-01"]
alfa = pd.read_csv("C:/Users/Phili/Desktop/fond/data/ALFA.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
alfa = alfa["Closing price"]#["2016-01-01":"2018-09-01"]


data = pd.DataFrame({"sand":sand, "eric":eric, "volv":volv, "hm":hm, "alfa":alfa}, index=sand.index)

"""
LÄNKAR:
    https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ddpg.py
    https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
    https://arxiv.org/pdf/1706.10059.pdf
    https://arxiv.org/abs/1509.02971

"""

print(len(data))
def main():
    batch_size = 32
    episodes = 1000
    max_steps = 200000
    env = PortfolioEnv(data, steps=256, trading_cost=0.0, time_cost=0.0, augment=0.1)
    sess = tf.Session()
    state_size = env.observation_space.spaces["history"].shape
    action_size = (env.action_space.shape[0], 1, 1)
    agent = Agent2(state_size, action_size, env, sess)
    for e in range(episodes):
        cur_state = env.reset()
        for _ in range(max_steps):
            cur_state = cur_state.reshape(1, state_size[0], state_size[1] ,state_size[2])
            action = agent.act(cur_state)
            action = action.reshape(1,action_size[0],1,1)
            new_state, reward,info, done = env.step(action)
            new_state = new_state.reshape(1, state_size[0], state_size[1] ,state_size[2])
            agent.remember(cur_state, action, reward, new_state, done)
            cur_state = new_state
            if done:
                print("Episode: {}/{}, episode end value: {}, weights: {}".format(e+1, episodes, info["portfolio_value"], action.reshape(action_size[0],)))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
main()