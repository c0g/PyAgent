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
    #ret = gp.optimise_hyper()
    #if not ret.success:
    #    print(ret)
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
    return CS
def draw_agent(t,gp,CS):
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
    plt.contourf(x,y,R,CS.levels)
    plt.colorbar(CS, shrink=0.8, extend='both')


cov = SqExpARD()
Sigma_c = np.eye(3)/0.001
hyp = np.log(np.array([10,2,2,100]))
lik = np.array([-1])
gp = GaussianProcess(lik,hyp,cov)
Z1 = np.array([[-1.,0.,2.]])
sample(Z1,gp)
Z2 = np.array([[1.,0.,3.]])
sample(Z2,gp)
Z3 = np.array([[0.,1.,4.]])
sample(Z3,gp)
Z4 = np.array([[0.,-1.,5.]])
sample(Z4,gp)
plt.show()
for t in xrange(0,10):
    oldT = gp.Z[-1][-1]
    control = solve_for_c(expected_mean,gp,Sigma_c)
    Z = np.array([np.hstack((control,np.array(oldT+1)))]) + np.random.randn(3)*0.001
    gp.hyp=hyp
    sample(Z,gp)
    plt.clf()
    CS=draw_reward(oldT+1)
    draw_agent(oldT+1,gp,CS)
    plt.plot(Z.flatten()[0],Z.flatten()[1],'o')
    plt.draw()
    time.sleep(0.1)


    
