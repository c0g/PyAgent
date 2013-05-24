from matplotlib import pyplot as plt
import numpy as np
from PyGP.GP import GaussianProcess
from PyGP.cov.NormalARD import NormalARD
from World import Reward
from Agent import Agent
import time


reward = Reward()
lik = np.log(np.array([0.00001]))
hyp = np.log(np.array([1, 1, 10]))
cov = NormalARD()
gp = GaussianProcess(lik, hyp, cov)
gp2 = GaussianProcess(lik, hyp, cov)

sig =np.ones((3,)) * 0.001
sig2 = np.ones((3,)) * 0.01
start_z = np.array([[0., 0., 0.]])
agent = Agent(gp, reward, sig, start_z)
agent2 = Agent(gp2, reward, sig2, start_z)
fig = plt.figure()
zlim = (-10, 10)
for i in xrange(0, 100):
    agent.observe()
    agent.decide()
    agent.act()
    agent2.observe()
    agent2.decide()
    agent2.act()

    t = agent.gp.Z[-1].flatten()[-1]

    fig.clf()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    cs = reward.draw(zlim, t, ax1)
    agent.draw(zlim, t, ax2, cs)
    agent2.draw(zlim, t, ax3, cs)
    fig.savefig(str(i).zfill(4) + ".jpg")
