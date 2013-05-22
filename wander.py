import numpy as np
from PyGP.GP import GaussianProcess
from PyGP.cov.NormalARD import NormalARD
from World import Reward
from Agent import Agent


reward = Reward()
lik = np.log(np.array([0.00001]))
hyp = np.log(np.array([1, 1, 10]))
cov = NormalARD()
gp = GaussianProcess(lik, hyp, cov)

sig = 0.001
start_z = np.array([[0., 0., 0.]])
agent = Agent(gp, reward, sig, start_z)

for i in xrange(0, 100):
    agent.observe()
    agent.decide()
    agent.act()
    print(agent.z)
