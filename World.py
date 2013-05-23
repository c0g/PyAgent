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

    def draw(self, t):
        for i, (xl, yl, tl) in enumerate(zip(X, Y, t)):
            z = np.array([[xl, yl, tl]])
            R[i] = self.calc(z)
        R.shape = (50, 50)
        CS = plt.contourf(x, y, R, 500)
        plt.colorbar(CS, shrink=0.8)
        return CS
