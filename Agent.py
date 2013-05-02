import numpy as np
import scipy as sp
class Agent:
    def __init__(self,x,t,world,worldModel,dim):
        #Current location
        self.x = x #current position
        self.cLoc = x #control location
        self.t = t #time
        self.world = world #function expresssing the true reward function
        self.worldModel = worldModel #this is our GP
        self.moveSig = 0.01 #our uncertainty about future location, given a control signal
        self.Sigmainv = worldModel.hyp[0]*np.eye(dim) + np.diag([self.moveSig] * dim-1 + [0]) 
        return
    def plan(self):
        # Agent finds optimal move, and stores the expected utility of that
        cLoc = self.solve_for_c()
        self.cLoc = cLoc + 0.1*np.random.randn()
        return
    def execute(self):

        return

    def solve_for_c(self): #Solve to find optimal control, given constraints.
        Kdd = self.world.K
        fd = self.world.Y
        mult = np.linalg.solve(Kdd,fd)
        Xd = self.world.X
        deltaC = np.Inf;
        oldC = self.cLoc
        while deltaC < self.threshold:
            dist = Xd - oldC
            rbfs = np.exp(dist.T.dot(self.Sigmainv.dot(dist))) #Radial basis functions
            w_rbfs = Xd * rbfs #Weighted RBFs
            mu1 = self.Sigmainv * dist * 2*w_rbfs.dot(mult)/(2*dist)#update mu1 from c
            mu2 = self.Sigmainv * dist * 2*w_rbfs.dot(mult)#update mu2 from c
            mu = np.vstack((2*mu1*dist,mu2))
            c = self.Sigmainv * dist * (w_rbfs.dot(mult) / rbfs.dot(mult)) - mu#update c
            deltaC = np.norm(oldC - c)
            oldC = c
        return c

    def sample(self,sample):
        sample.shape=(1,1)
        if self.worldModel.X == None:
            self.worldModel.infer(x,sample)
        else:
            self.worldModel.infer_iter(x,sample)
        return
    def utility(self,x,t):
        return self.worldModel

    def move(self):#Decide whether to move, or sample based on expected utility from a move and expected utility from a sample

        return
    def find_cost(self):
        Kdd = self.world.K
        fd = self.world.Y
        mult = np.linalg.solve(Kdd,fd)
        Xd = self.world.X
        self.cost = lambda c: 0 if (np.abs(c) >= 10) else - np.dot(np.exp(-(c-self.l[0])**2).T,mult) - self.world.get_Ys2(c)
        return
