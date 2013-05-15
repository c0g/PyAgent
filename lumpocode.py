# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>
from scipy import optimize as opt
import numpy as np
import sys
from PyGP.GP import GaussianProcess
from PyGP.cov.SquaredExponentialARD import SqExpARD
import pylab as pb
pb.ion()
from matplotlib import pyplot as plt

def expected_mean(control_aug,Sigma_c,gp):
    '''
    House keeping with the control input. It is comprised of [x,y,z] that is the spatial location
    We need to add the [fixed] time into a control vector
    '''
    Zn = Znow.flatten()
    t = Zn[-1] + 1
    c_aug_flat = control_aug.flatten()
    control = np.hstack((c_aug_flat, np.array([t])))
    Sigma = Sigma_c + Sigma_gp

    def rbfs(zd):  return np.exp(-(zd-control).T.dot(np.linalg.solve(Sigma,zd-control)))
    def drbfs(zd): return -2*np.linalg.solve(Sigma,control-zd) * rbfs(zd)
    
    f = np.sum([rbfs(zd)*w for  zd,w in zip(Zd,Kddifd)])
    df = np.sum(np.array([drbfs(zd) for zd  in Zd]),0) 
    return (f,df[:-1])

def expected_variance(control_aug,Sigma_c,gp):
    """Our current location is the last entry in the GP."""
    Znow = gp.Z[-1,:]
    Sigma_gp = np.diag(np.exp(2*gp.hyp[1:]))
    Zd = gp.Z[0][-1]
    Kdd = gp.K

    assert Zd.ndim == 2
    Zn = Znow.flatten()
    t = Zn[-1] + 1
    c_aug_flat = control_aug.flatten()
    control = np.hstack((c_aug_flat, np.array([t])))
    Kmine =  cov.K(hyp,np.array([control]),np.array([control]))[0] /  np.sqrt( np.linalg.det(Sigma_c)*(2*np.pi)**3 )
    
    KddI = np.linalg.inv(Kdd)
    Sigma_v = np.linalg.inv((2*Sigma_c+2*Sigma_gp))
    scale = np.sqrt(1/(2*np.linalg.det(1+np.linalg.solve(Sigma_gp,Sigma_c))))
    def zself(zdi,zdj):  return np.exp(-(zdi+zdj).T.dot(2*np.linalg.inv(Sigma_gp)).dot(zdi+zdj)) 
    def cz(control,zdi,zdj): return np.exp(-(control - (zdi+zdj)/2).T.dot(np.linalg.inv(Sigma_v)).dot(control - (zdi+zdj)/2))
    def dc(control,zdi,zdj): return -2 * (control - (zdi+zdj)/2).T.dot(np.linalg.inv(Sigma_v))
    samples = np.shape(Zd)[0]
    sumf = 0
    sumdf = 0
    for i in xrange(0,samples):
        for j in xrange(0,samples):
            sumf += cz(control,Zd[i,:],Zd[j,:]) * zself(Zd[i,:],Zd[j,:]) * KddI[i,j]
            sumdf += dc(control,Zd[i,:],Zd[j,:]) * cz(control,Zd[i,:],Zd[j,:]) * zself(Zd[i,:],Zd[j,:]) * KddI[i,j]
    return(Kmine - scale*sumf,-scale*sumdf[:-1]) 

def test_grads():
    from scipy.optimize import check_grad
    """These functions are defined for a GP, so we need to define a GP to test them over"""
    cov = SqExpARD()
    D = 2
    hyp = np.array([1.,1.,1.])
    lik = np.array([0.01])
    gp = GaussianProcess(lik,hyp,cov)
    """Make a fake observation, just to constrain the GP..."""
    gp.observe(np.array([1.,1.]),np.array([1]))
    """Now draw a few values, then push them straight back into the GP"""
    Z1 = np.array([2.,2.])
    gp.observe(Z1,gp.draw(Z1))
    Z2 = np.array([0.,0.])
    gp.observe(Z1,gp.draw(Z2))

    control = np.array([0.5,1.5])

    def fm(control): return expected_mean(control,Sigma_c,gp)[0]
    def dfm(control): return expected_mean(control,Sigma_c,gp)[1]
    check_grad(fm,dfm,control)




scales = np.random.randn(20,1)*1
centers = np.random.randn(20,2)*4
signs = np.sign(np.random.randn(20,1))
rates = np.random.randn(20,1)*0.1
def reward(z):
    t = z.flatten()[-1]
    x = z.flatten()[:-1]
    f = 0
    for rate,sign,scale,center in zip(rates,signs,scales,centers):
        f += sign.flatten()*np.exp(-(center - x).T.dot(center - x)/10)
    reward = f
    return reward
def draw_reward(t):
    x=y=linspace(-10,10,40)
    (X,Y) = np.meshgrid(x,y)
    X.shape=Y.shape=(40**2,1)
    R = np.zeros((40**2,1))
    for i,(x_el,y_el) in enumerate(zip(X,Y)):
        z = np.array([x_el,y_el,t])
        R[i] = reward(z)
    R.shape=(40,40)
    CS = plt.contourf(x,y,R)
    plt.colorbar(CS, shrink=0.8, extend='both')

def wrap_mean(mean_func,Zd,Sigma_c,Sigma_gp,Znow,Kddifd):
    def mean(control): return mean_func(control,Zd,Sigma_c,Sigma_gp,Znow,Kddifd)
    return mean
def wrap_var(var_func,Zd,Sigma_c,Sigma_gp,Znow,Kdd,cov,hyp):
    def var(control): return var_func(control,Zd,Sigma_c,Sigma_gp,Znow,Kdd,cov,hyp)
    return var
def disutility(mean,variance):
    """mean and variance are closures containing all the values they need, except for the control input
    since we're using a linear disutility [sigh] the derivative is a linear function of the two derivatives
    We get two functions back from this, giving the expected value of the utility and it's derivative"""
    def f(control):
        return (mean(control)[0] )#+ variance(control)[0])
    def df(control):
        return (mean(control)[1] )#+ variance(control)[1])
    return(f,df)

def sample(z):
    r = reward(z)
    print("Reward: " + str(r) + " at " + str(z))
    gp.observe(z,r)
    ret = gp.optimise_hyper()
    if not ret.success:
        print(ret)
    return

def next_move(f,df,Znow):
    """f, df are the function and derivatives of the disutility"""
    def cons_x(control,Znow):
        x = Znow.flatten()[:-1]
        dist = np.linalg.norm(control - x)
        return 1 - dist**2
    def dcons_x(control,Znow):
        x = Znow.flatten()[:-1]
        return  - 2 * (control - x)

    eq_con = { 'type':'eq',
        'fun':cons_x,
        'dfun':dcons_x,
        'args':[Znow] }
    ret = opt.minimize(f,Znow,method='CG',jac=df)
    if not ret.success:
        print(ret)
    return(ret.x)

# <codecell>


cov = SqExpEuc()
hyp = np.array([-0.1,-7])
lik = np.array([-0.1])
gp = GaussianProcess(lik,hyp,cov)
Sigma_c = np.eye(3)*0.01
def Sigma_gp(gp,d):
    return np.eye(d) * np.exp(-2*gp.hyp[0])

def test(): 
    
    Znow = np.array([[0.,0.,0.]])
    sample(Znow)
    Znow = np.array([[1.,1.,1.]])
    sample(Znow)
    for it in xrange(0,12):
        (f,df) = disutility(Sigma_c,Znow,gp)
        next_x = next_move(f,df,Znow)
        Znow = np.hstack((next_x,Znow.flatten()[-1]+1))+np.hstack((np.random.randn(2)*0.01,np.array([0])))
        sample(np.array([Znow]))
        plt.clf()
        draw_reward(it+2)
        plt.plot(Znow.flatten()[0],Znow.flatten()[1],'o')
        plt.show()
