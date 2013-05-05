import numpy as np
import scipy.optimize as opt
def solve_for_c(Sigmainv,Kdd,fd,Zd,Znow): #Solve to find optimal control, given constraints.
    print("Znow start of opt",Znow)
    mult = np.linalg.solve(Kdd,fd)
    def t_con(control):
        return((control[-1] - (1 +  Znow[-1])))
    def t_dcon(control):
        z = np.zeros(np.shape(control))
        z[-1] = 1
        return z
    def x_con(control):
        return np.linalg.norm(control.flatten()[:-1] - Znow.flatten()[:-1] - 0.1)
    def x_dcon(control):
        sel = np.eye(np.shape(control)[0])
        sel[-1] = 0
        return 2*(control.flatten()-Znow.flatten()).T.dot(sel)
    
    def cost(control):
        control[-1] = Znow[0][-1] + 1
        rbfs = np.array([(-(Zdj.flatten()-control.flatten()).T.dot(Sigmainv.dot(Zdj.flatten()-
            control.flatten()))) * weight for Zdj,weight in zip(Zd,mult)])
        f = np.sum(rbfs)
        return(f.flatten())
    constraints = [{'type':'ineq','fun':x_con, 'jac':x_dcon}]
    print("Znow start of opt",Znow)
    ret = opt.minimize(cost,Znow,method='COBYLA',options={'maxiter':10000})
    control = ret.x
    print(ret)
    control[-1] = Znow[0][-1] + 1
    print("Control out of loop",control)
    print("Znow out of loop",Znow)
    control.shape = (1,3)

    return control
    
