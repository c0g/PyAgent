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
#This is the the covariance over the Hazard functions
#We use a squared exponential GP to represent our hazard function
cov  = Plus(EuclideanDist(SqExp()),Noise(10))
nhyp = cov.hyp(1)
hyp = np.array([-0.1,1.])
hazard_gp = GaussianProcess(hyp,cov)
#Luckily our hazard function is a sqexp centered at x=2
utility = lambda x: 50*np.exp(-(x-2)**2) - 2 * np.exp(-(x+5)**2 if np.abs(x) < 10 else 0)# + 2 * np.exp(-(x+1.5)**2)
utility_vec = lambda x: [utility(xl) for xl in x]
x = np.linspace(-10,10,1000)
x.shape = (1000,1)
plt.ion()
print(plt.isinteractive())
agent = Agent(0,hazard_gp)
plt.clf()
i=0
agent.sample(utility(agent.x))
agent.x += 0.1
agent.sample(utility(agent.x))
while i<100:
    agent.sample(utility(agent.x))
    lastx = agent.x
    agent.plan()
    agent.move()
    thisx = agent.x
    print(agent.x)
    if not np.mod(i,1):
        plt.hold(True)
        agent.world.predict(x)
        p1, = plt.plot(x,utility_vec(x))
        p2, = plt.plot(x,agent.world.Ymu)
        p3, = plt.plot(x,agent.world.Ymu+2*np.sqrt(agent.world.Ys2))
        p4, = plt.plot(x,agent.world.Ymu-2*np.sqrt(agent.world.Ys2))
        p5, = plt.plot(agent.world.X,agent.world.Y,'o')
        p6, = plt.plot(agent.x,utility(agent.x),'o')
        plt.legend([p1,p2,p3,p4,p5,p6],["Hazard","Agent Mean","+2 \sigma","-2 \sigma","Past Locations","Current Location"])
        plt.savefig("images/" + str(i).zfill(3) + ".jpg")
        plt.clf()
    i+=1
print agent.x
agent.world.predict(x)
plt.hold(True)
p1, = plt.plot(x,utility_vec(x))
p2, = plt.plot(x,agent.world.Ymu)
p3, = plt.plot(x,agent.world.Ymu+2*np.sqrt(agent.world.Ys2))
p4, = plt.plot(x,agent.world.Ymu-2*np.sqrt(agent.world.Ys2))
p5, = plt.plot(agent.world.X,agent.world.Y,'o')
p6, = plt.plot(agent.x,utility(agent.x),'o')
plt.legend([p1,p2,p3,p4,p5,p6],["Hazard","Agent Mean","+2 \sigma","-2 \sigma","Past Locations","Current Location"])
plt.savefig("images/" + str(i).zfill(3) + ".jpg")
