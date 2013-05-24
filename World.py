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

    def draw(self, zlim, t, ax, cs=None):
        xmin = zlim[0]
        xmax = zlim[1]
        ymin = zlim[2]
        ymax = zlim[3]
        x = np.linspace(xmin, xmax)
        y = np.linspace(ymin, ymax)
        (X, Y) = np.meshgrid(x, y)
        X.shape = Y.shape = (50**2, 1)
        T = np.ones((50**2, 1)) * t
        R = np.zeros((50**2, 1))
        for i, (xl, yl, tl) in enumerate(zip(X, Y, T)):
            z = np.array([[xl, yl, tl]])
            R[i] = self.calc(z)
        R.shape = (50, 50)
        if cs:
            CS = ax.contourf(x, y, R, cs.levels)
        else:
            CS = ax.contourf(x, y, R, 500)
        return CS
