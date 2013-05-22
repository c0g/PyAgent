from matplotlib import pyplot as plt
import numpy as np
class ViewPort(object):
    def __init__(self):
        self.cs = None

    def draw(self, func, time, position):
        """Uses the given callable to draw a contour plot
        Callable must accept an NxD array of numbers and return
        a Nx1 array of results"""
        x = y = np.linspace(-10, 10, 50)
        (X, Y) = np.meshgrid(x, y)
        X.shape = Y.shape = (50**2, 1)
        R = np.zeros((50**2, 1))
        T = np.ones(np.shape(X)) * time
        ZPred = np.hstack((X, Y, T))
        R = func(ZPred)
        plt.contourf(x, y, R)
