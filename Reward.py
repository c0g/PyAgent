import numpy as np


class Reward(object):

    def __init__(self):
        self.scales = np.random.randn(20, 1)*1
        self.centers = np.random.randn(20, 2)*4
        self.signs = np.sign(np.random.randn(20, 1))
        self.rates = np.random.randn(20, 1)*0.1

    def calc(self,z):
        t = z.flatten()[-1]
        x = z.flatten()[:-1]
        f = 0
        for rate, sign, scale, center in zip(self.rates, self.signs, self.scales, self.centers):
            f += sign.flatten()*np.exp(-(center - x).T.dot(center - x)/10) * np.sin(rate*t)
        reward = f
        return(reward)
