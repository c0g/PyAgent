import pylab as pb
pb.ion()
import time
from matplotlib import pyplot as plt
import numpy as np
from PyGP.GP import GaussianProcess
from PyGP.cov.SquaredExponentialARD import SqExpARD
from solve_for_c import solve_for_c
from expectations import expected_mean
scales = np.random.randn(20,1)*1
centers = np.random.randn(20,2)*4
signs = np.sign(np.random.randn(20,1))
rates = np.random.randn(20,1)*0.1
def reward(z):
    t = z.flatten()[-1]
    x = z.flatten()[:-1]
    f = 0
    for rate,sign,scale,center in zip(rates,signs,scales,centers):
        f += sign.flatten()*np.exp(-(center - x).T.dot(center - x)/10) #* np.sin(10*np.cos(rate*t))
    reward = f
    return reward
def sample(z,gp):
    r = reward(z)
    print("Reward: " + str(r) + " at " + str(z))
    gp.observe(z,np.array([r]))
    ret = gp.optimise_hyper()
    if not ret.success:
        print(ret)
    return
def draw_reward(t):
    x=y=np.linspace(-10,10,30)
    (X,Y) = np.meshgrid(x,y)
    X.shape=Y.shape=(30**2,1)
    R = np.zeros((30**2,1))
    for i,(x_el,y_el) in enumerate(zip(X,Y)):
        z = np.array([x_el,y_el,t])
        R[i] = reward(z)
    R.shape=(30,30)
    plt.subplot(121)
    CS = plt.contourf(x,y,R)
    plt.colorbar(CS, shrink=0.8, extend='both')
def draw_agent(t,gp):
    x=y=np.linspace(-10,10,30)
    (X,Y) = np.meshgrid(x,y)
    X.shape=Y.shape=(30**2,1)
    R = np.zeros((30**2,1))
    for i,(x_el,y_el) in enumerate(zip(X,Y)):
        z = np.array([[x_el,y_el,t]])
        gp.predict(z)
        R[i] = gp.Ymu
    R.shape=(30,30)
    plt.subplot(122)
    CS = plt.contourf(x,y,R)
    plt.colorbar(CS, shrink=0.8, extend='both')


cov = SqExpARD()
Sigma_c = np.eye(3)*0.1
Sigma_c[-1] = 0
hyp = np.array([-1,-1,-1,10])
lik = np.array([-10])
gp = GaussianProcess(lik,hyp,cov)
"""Make a fake observation, just to constrain the GP..."""
Z1 = np.array([[2.,2.,2.]])
sample(Z1,gp)
Z2 = np.array([[2.,1.,3.]])
sample(Z2,gp)
def utility(ctl,sig,gp): -expected_mean(ctl,sig,gp)

for t in xrange(0,10):
    oldT = gp.Z[-1][-1]
    control = solve_for_c(utility,gp,Sigma_c)
    Z = np.array([np.hstack((control+np.random.randn(2)*0.1,np.array(oldT+1)))]) 
    sample(Z,gp)
    plt.clf()
    draw_reward(oldT+1)
    draw_agent(oldT+1,gp)
    plt.plot(Z.flatten()[0],Z.flatten()[1],'o')
    plt.draw()
    time.sleep(0.1)


    
