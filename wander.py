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
sig2 = np.ones((3,)) * 0.1
start_z = np.array([[0., 0., 0.]])
agent = Agent(gp, reward, sig, start_z)
agent2 = Agent(gp2, reward, sig2, start_z)
fig = plt.figure(figsize=(20,7), dpi=300)
zlim = (-10, 10, -10, 10)
for i in xrange(0, 1000):
    agent.observe()
    agent.decide()
    agent.act()
    agent2.observe()
    agent2.decide()
    agent2.act()

    t = agent.gp.Z[-1].flatten()[-1]
    a = [0] * 4
    a[0] = agent.gp.Z[-1].flatten()[0]
    a[1] = agent.gp.Z[-1].flatten()[1]
    a[2] = agent.gp.Z[-1].flatten()[0]
    a[3] = agent.gp.Z[-1].flatten()[1]
    extent = np.max(np.abs(a))
    lim = extent + 3 if extent > 10 else 10
    zlim = (-lim, lim, -lim, lim)
    fig.clf()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_aspect('equal')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_aspect('equal')

    cs = reward.draw(zlim, t, ax1)
    agent.draw(zlim, t, ax2, cs)
    agent2.draw(zlim, t, ax3, cs)
    fig.savefig("./images/" + str(i).zfill(4) + ".jpg")
