import numpy as np
import scipy.optimize as opt
def solve_for_c(Sigmainv,Kdd,fd,Zd,Znow,utility): #Solve to find optimal control, given constraints.
    print("Znow start of opt",Znow)
    mult = np.linalg.solve(Kdd,fd)
    def t_con(control):
        return((control[-1] - (1 +  Znow[-1])))
    def t_dcon(control):
        z = np.zeros(np.shape(control))
        z[-1] = 1
        return z
    def x_con(control):
        return np.linalg.norm(control.flatten()[:-1] - Znow.flatten()[:-1]) - 0.1
    def x_dcon(control):
        sel = np.eye(np.shape(control)[0])
        sel[-1] = 0
        return 2*(control.flatten()-Znow.flatten()).T.dot(sel)
    
    def cost(control):
        control[-1] = Znow[0][-1] + 1
        Ks = np.array([(-(Zdj.flatten()-control.flatten()).T.dot(Sigmainv.dot(Zdj.flatten()-
            control.flatten()))) for Zdj in Zd])
        Ymu = np.dot(Ks.T,mult)
        Ys2 = np.dot(Ks,np.linalg.solve(Kdd,Ks))
        f = utility(Ymu,Ys2,control)
        return(f.flatten())
    constraints = [{'type':'ineq','fun':x_con, 'jac':x_dcon}]
    tbound = Znow.flatten()[-1]
    bounds= [(None,None),(None,None),(tbound,tbound)]
    ret = opt.minimize(cost,Znow,method='CG',constraints=constraints,bounds=bounds,options={'maxiter':10000})
    if not ret.success:
        print ret
    control = ret.x.flatten()
    control[-1] = Znow[0][-1] + 1
    control.shape = (1,3)

    return control
    
