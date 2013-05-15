import numpy as np
import scipy.optimize as opt
from expectations import 


def cost_func(control_aug,t,Zd,Sigmainv,Znow):

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





