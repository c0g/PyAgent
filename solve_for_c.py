import numpy as np
import scipy.optimize as opt
from expectations import expected_mean
from PyGP.GP import GaussianProcess
from PyGP.cov.SquaredExponentialARD import SqExpARD
def solve_for_c(utility,gp,Sigma_c): #Solve to find optimal control, given constraints.
    Znow = gp.Z[-1].flatten()
    def con(control,Znow):
         x = Znow.flatten()[:-1]
         dist = np.linalg.norm(control - x)
         return 1 - dist**2
    def dcon(control,Znow):
         x = Znow.flatten()[:-1]
         return  - 2 * (control - x)

    eq_con = { 'type':'eq',
         'fun':con,
         'dfun':dcon,
         'args':[Znow] } 
    def f(control):return expected_mean(control,Sigma_c,gp)[0]
    def df(control): return expected_mean(control,Sigma_c,gp)[1]
    ret = opt.minimize(f,Znow[:-1],jac=df,method='SLSQP',constraints=eq_con)
    if not ret.success:
        print ret
    return(ret.x.flatten())

if __name__=="__main__":
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
    def utility(ctl,sig,gp): -expected_mean(ctl,sig,gp)
    print(solve_for_c(utility,gp,Sigma_c))
