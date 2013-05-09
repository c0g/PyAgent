import numpy as np
import scipy.optimize as opt
def cost(control,Sigmainv,Zd):
    return



def cost_func(control_aug,t,Zd,Sigmainv,Znow):
    '''
    House keeping with the control input. It is comprize of [x,y,z....mu1,mu2] that is the spatial location + lagrange mutlipliers.
    We need to remove the lagranges, add the [fixed] time into a control vector
    '''
    c_aug_flat = control_aug.flatten()
    mu = c_aug_flat[-1]
    control = c_aug_flat
    control[-1] = t #Replace mu with t

    Zn = Znow.flatten()
    
    space_sel = np.eye(control.shape[0])
    space_sel[-1] = 0

    rbfs = lambda zd: np.exp(-(zd-control).T.dot(Sigmainv).dot(zd-control))#This is probably slow...
    drbfs = lambda zd: -2*Sigmainv.dot(control-zd) * rbfs(zd)
    step_constraint = np.exp(-(control-Zn).T.dot(space_sel).dot(control-Zn)) - 1
    f = -np.sum([rbfs(zd.T) for zd in Zd]) + mu * step_constraint
    df = -np.sum(np.array([drbfs(zd.T) for zd in Zd]),0) -2 * mu * space_sel.dot(control-Zn)

    return (f,df)

if __name__=="__main__":
    Zd = np.array([[1,0,0]])
    Sigmainv = np.eye(3)
    control = Znow = np.array([[0,0,0]])
    t = 0
    f = lambda control,t,Zd,Sigmainv,Znow: cost_func(control,t,Zd,Sigmainv,Znow)[0]
    df = lambda control,t,Zd,Sigmainv,Znow: cost_func(control,t,Zd,Sigmainv,Znow)[1]
    print(opt.check_grad(f,df,control,t,Zd,Sigmainv,Znow))
    ret=opt.minimize(f,control,jac=df,method='CG',args=(t,Zd,Sigmainv,Znow))
    print(ret)
    retnj=opt.minimize(f,control,method='CG',args=(t,Zd,Sigmainv,Znow))
    print(retnj)





