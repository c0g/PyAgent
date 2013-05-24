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
gp2 = GaussianProcess(lik, hyp, cov)

sig =np.ones((3,)) * 0.001
sig2 = np.ones((3,)) * 0.001
start_z = np.array([[0., 0., 0.]])
agent = Agent(gp, reward, sig, start_z)
agent2 = Agent(gp2, reward, sig2, start_z)

for i in xrange(0, 100):
    agent.observe()
    agent.decide()
    agent.act()
    agent2.observe()
    agent2.decide()
    agent2.act()
    print("Agent 1: " + str(agent.z))
    print("Agent 2: " + str(agent2.z))
