import numpy as np
def solve_for_c(threshold,Sigmainv,Kdd,fd,Zd,oldC,Znow): #Solve to find optimal control, given constraints.
    mult = np.linalg.solve(Kdd,fd)
    deltaC = np.Inf
    control = oldC
    mu1sel = np.eye(np.shape(Zd)[1])
    mu1sel[-1] = 0
    mu2sel = np.zeros((1,np.shape(Zd)[1]))
    mu2sel[-1] = 1
    loop = 1
    while deltaC > threshold:
        loop+=1
        mu1 = (Znow-control).T.dot(mu1sel).dot(Znow-control) + 1
        mu2 = mu2sel.dot((Znow-control)) + 1
        rbfs = np.array([np.exp(-(Zdj.flatten()-control.flatten()).T.dot(Sigmainv.dot(Zdj.flatten()-control.flatten()))) * weight for Zdj,weight in zip(Zd,mult)]) #Radial basis functions

        zrbfs =np.array([Zdj.T.dot(Sigmainv) * rbf for rbf,Zdj in zip(rbfs,Zd)]) #rbfs times by ZdSiginv
        premult = 2*Sigmainv * np.sum(rbfs) + 2*mu1sel
        postmult = 2*np.sum(zrbfs,0) + 2*mu1*Znow.T.dot(mu1sel) + mu2*mu2sel
        oldC = control
        control = np.linalg.solve(premult,postmult.T)
        deltaC = np.linalg.norm(oldC - control)
        print(control)
    return control

if __name__ == "__main__":
    #Generate some fake data
    Zd = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]);
    fd = np.array([[1],[5],[0],[-1]])
    Kdd = np.eye(4)
    oldc = np.array([[0],[0]])
    Znow = np.array([[0],[0]])
    Sigmainv = np.eye(2)
    print(solve_for_c(0.00001,Sigmainv,Kdd,fd,Zd,oldc,Znow))
