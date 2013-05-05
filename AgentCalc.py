from GP import GaussianProcess
from cov.SquaredExponential import SqExp
from cov.Noise import Noise
from cov.dist.Euclidean import EuclideanDist
from cov.meta.Plus import Plus
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from Agent import Agent
import time
from Reward import Reward
import sys
if globals().has_key('init_modules'):
    for m in [x for x in sys.modules.keys() if x not in init_modules]:
        del(sys.modules[m])
else:
    init_modules = sys.modules.keys()

#This is the the covariance over the Hazard functions
#We use a squared exponential GP to represent our hazard function
cov  = Plus(EuclideanDist(SqExp()),Noise(10))
nhyp = cov.hyp(1)
hyp = np.array([-0.1,-0.1,1.])
worldModel = GaussianProcess(hyp,cov)

#Luckily our hazard function is a sqexp centered at x=2
centers = np.array([[1,1],[2,2]])
magnitudes = np.array([1,2])
speeds = np.array([0.01,0.01])
frequencies = np.array([0.1,0.1])
world = Reward(centers,magnitudes,speeds,frequencies)

#Generate a 2d meshgrid
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
(X,Y) = np.meshgrid(x,y)
X.shape = Y.shape = (10000,1)
utility = lambda Ymu,Ys2,Z: (Ymu+Ys2**2)
agent = Agent(np.array([[0,0,0]]),world,worldModel,utility)
agent.sample()
agent.Z = np.array([[1,0,1]])
agent.sample()
i = 0
states = [np.zeros(3)]*10
state = 0
while state<10:
    agent.plan()
    agent.execute()
    states[state] = agent.Z
    state+=1
