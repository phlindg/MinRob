import pandas as pd
import numpy as np
import gym

class DataSrc:
    """
    tror ej denna behövs
    """
    def __init__(self, df, steps=252, augment=0.0, window_length = 50):

        self.steps = steps+1
        self.augment = augment
        self.window_length = window_length
        
        df = df.copy()
        df.replace(np.nan, 0, inplace=True)
        df = df.fillna(method="pad")
        
        self.asset_names = df.columns.tolist()
        self.features = 1
        self.data = None
        self.step = None

        data = df.values.reshape(len(df), len(self.asset_names), self.features)#ettan är antal features <-- bara pris, hmm.
        self._data = np.transpose(data, (1,0,2))
        self._times = df.index



        self.reset()
    def _step(self):
        data_window = self.data[:,self.step:(self.step + self.window_length)].copy()

        y1 = data_window[:, -1, 0]/data_window[:, -2, 0]

        rets = np.diff(np.log(data_window + 1e-7), axis=1)
        rets = np.insert(rets, 0, 0,axis=1)
        self.step += 1
        history = rets
        done = self.step >= self.steps

        return history, y1, done
    def reset(self):
        self.step = 0

        self.idx = np.random.randint(low = self.window_length, high=self._data.shape[1]-self.steps)
        data = self._data[:,(self.idx - self.window_length):(self.idx + self.steps+1)]

        self.times = self._times[(self.idx - self.window_length):(self.idx + self.steps+1)]

        #prevent overfitting
        data += np.random.normal(loc=0, scale=self.augment, size=data.shape)

        self.data = data


class PortfolioSim:
    def __init__(self, asset_names=[], steps=128, trading_cost = 0.0025, time_cost = 0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.asset_names = asset_names
        self.steps = steps

        self.w0 = None
        self.p0 = None
        self.infos = []

        self.reset()
    def step(self, w1, y1):

        w0 = self.w0
        p0 = self.p0

        #eq 7
        #y1 = y1.reshape(1,3,1,1)
        w0 = w0.reshape(5,)
        dw1 = (y1*w0)/(np.dot(y1,w0) + 1e-7)

        #eq 16 - cost to change portfolio
        c1 = self.cost * (np.abs(dw1 - w1)).sum()

        #eq 11 - final portfolio value
        p1 = p0 * (1-c1)*np.dot(y1,w0)
        #cost of holding
        p1 = p1*(1-self.time_cost) # <-- varför skulle vi ha det.

        p1 = np.clip(p1, 0, np.inf)

        rho1 = p1/p0 - 1
        r1 = np.log((p1+1e-7)/(p0 + 1e-7))
        reward = r1/self.steps # eq22 - immediate reward scalas med episode length
        #reward = p1 - p0
        self.w0 = w1
        self.p0 = p1

        done = p1 == 0

        info = {
            "reward":reward,
            "portfolio_value": p1,
            "cost":c1,
            "market_return":y1.mean()
        }
        #for i, name in enumerate(self.asset_names):
        #    info["weight_"+name] = w1[:,i,:,:]
        #    info["price_"+name] = y1[i]
        self.infos.append(info)
        return reward, info, done
    def reset(self):
        self.infos= []
        self.w0 = np.array([1.0] + [0.0]*(len(self.asset_names)-1))
        self.p0 = 1.0

class PortfolioEnv(gym.Env):
    def __init__(self, df, steps=252, trading_cost = 0.0025, time_cost = 0.0, window_length=50, augment=0.0):
        
        self.src = DataSrc(df = df, steps = steps, augment = augment, window_length=window_length)
        self.sim = PortfolioSim(self.src.asset_names, trading_cost=trading_cost, time_cost=time_cost, steps=steps)

        self.infos = []
        
        n_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=(n_assets,))
        obs_shape = (
            n_assets, window_length, self.src.features
        )
        self.observation_space = gym.spaces.Dict({
            "history": gym.spaces.Box(
                -10, 1, obs_shape
            ),
            "weights": self.action_space
        })
        self.reset()
    def step(self, action):
        #print("action: ", action)
        
        action = action.reshape(5,)
        #weights = np.clip(action, -1.0, 1.0)
        weights = action
        weights /= weights.sum() + 1e-7
        assert self.action_space.contains(action)
        #np.testing.assert_almost_equal(np.sum(weights), 0.0, 3)

        history, y1, done1 = self.src._step()
        #print("HISTORY: ", history)
        reward, info, done2 = self.sim.step(weights, y1)

        info["market_value"] = np.cumprod(
            [i["market_return"] for i in self.infos + [info]]
        )[-1]
        info["date"] = self.src.times[self.src.step].timestamp()
        info["steps"] = self.src.step

        self.infos.append(info)

        return history, reward, info, done1 or done2
    def reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        obs, reward, info, done = self.step(action)
        return obs
    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]
        


        
