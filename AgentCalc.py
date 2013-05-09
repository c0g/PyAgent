from PyGP.GP import GaussianProcess
from PyGP.cov.SquaredExponentialEuc import SqExpEuc
from PyGP.cov.Noise import Noise
from PyGP.cov.dist.Euclidean import EuclideanDist
from PyGP.cov.meta.Plus import Plus
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

world = GaussianProcess(hyp,cov)
xobs = np.array([[0,0,0]])
fobs = np.array([[0]])
world.infer(xobs,fobs)

#Generate a 2d meshgrid
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
(X,Y) = np.meshgrid(x,y)
X.shape = Y.shape = (10000,1)
utility = lambda Ymu,Ys2,Z: Ymu - np.linalg.norm(Z[:-1])
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
