from scipy import optimize as opt
import numpy as np
import sys
from PyGP.GP import GaussianProcess
from PyGP.cov.SquaredExponentialARD import SqExpARD
import pylab as pb
pb.ion()

def expected_mean(control_aug,Sigma_c,gp):
    Znow = gp.Z[-1]
    Zd = gp.Z
    ell = np.exp(-2*gp.hyp[1:])
    Sigma_gp = np.diag(ell)
    Kddifd = np.linalg.solve(gp.cov.K(gp.hyp,gp.Z,gp.Z)[0],gp.F)
    Zn = Znow.flatten()
    t = Zn[-1] + 1
    c_aug_flat = control_aug.flatten()
    control = np.hstack((c_aug_flat, np.array([t])))
    Sigma = Sigma_c + Sigma_gp

    def rbfs(zd):  return np.exp(-(zd-control).T.dot(np.linalg.solve(Sigma,zd-control)))
    def drbfs(zd): return -2*np.linalg.solve(Sigma,control-zd) * rbfs(zd)
    
    f = np.sum([rbfs(zd)*w for  zd,w in zip(Zd,Kddifd)])
    df = np.sum(np.array([drbfs(zd) for zd  in Zd]),0) 
    return (f.flatten(),df[:-1].flatten())

def expected_variance(control_aug,Sigma_c,gp):
    """Our current location is the last entry in the GP."""
    Znow = gp.Z[-1,:]
    ell = np.exp(-2*gp.hyp[1:])
    Sigma_gp = np.diag(ell)
    Zd = gp.Z[0][-1]
    Kdd = gp.K
    Zn = Znow.flatten()
    t = Zn[-1] + 1
    c_aug_flat = control_aug.flatten()
    control = np.hstack((c_aug_flat, np.array([t])))

    Znow = gp.Z[-1]
    Zd = gp.Z
    ell = np.exp(gp.hyp[1:])
    Sigma_gp = np.diag(ell*2)    
    Zn = Znow.flatten()
    t = Zn[-1] + 1
    c_aug_flat = control_aug.flatten()
    control = np.hstack((c_aug_flat, np.array([t])))

    Kmine =  gp.cov.K(gp.hyp,np.array([control]),np.array([control]))[0] #/  np.sqrt( np.linalg.det(Sigma_c)*(2*np.pi)**3 )
    dKmine =  gp.cov.K(gp.hyp,np.array([control]),np.array([control]))[1] #/  np.sqrt( np.linalg.det(Sigma_c)*(2*np.pi)**3 )

    
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

    print(Kmine,sumf)
    print(sumdf[:-1])
    return(Kmine - scale*sumf,dKmine-scale*sumdf[:-1]) 

def test_grads():
    from scipy.optimize import check_grad
    """These functions are defined for a GP, so we need to define a GP to test them over"""
    cov = SqExpARD()
    Sigma_c = np.eye(3)*0.1
    Sigma_c[-1] = 0
    hyp = np.array([1.,.9,2.,1.])
    lik = np.array([0.01])
    gp = GaussianProcess(lik,hyp,cov)
    """Make a fake observation, just to constrain the GP..."""
    gp.observe(np.array([[1.,1.,1.]]),np.array([1]))
    """Now draw a few values, then push them straight back into the GP"""
    Z1 = np.array([[2.,2.,2.]])
    gp.observe(Z1,gp.draw(Z1))
    Z2 = np.array([[0.,0.,3.]])
    gp.observe(Z2,gp.draw(Z2))
    dist = 0.0001
    control = np.array([1.5,1.5])
    def fm(control): return expected_mean(control,Sigma_c,gp)[0]
    def dfm(control): return expected_mean(control,Sigma_c,gp)[1]
    print(check_grad(fm,dfm,control))
    """
    def fv(control): return expected_variance(control,Sigma_c,gp)[0]
    def dfv(control): return expected_variance(control,Sigma_c,gp)[1]
    print(check_grad(fv,dfv,control))
    """

