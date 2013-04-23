import numpy as np
import scipy as sp
class Agent:
    def __init__(self,x,world):
        self.x = x
        self.cLoc = 0
        self.world = world
        self.cost = lambda c: 0
        self.d = 0
        self.t = 0;
        return
    def plan(self):
        #Find minimal cost w.r.t new centroid
        self.cLoc = sp.optimize.brute(self.cost,[(self.x-0.1, self.x+0.1),(self.t,self.t+1)])
        self.t = self.cLoc[1]
        self.cLoc[0] = self.cLoc[0] + 0.03*np.random.randn()
        self.d = self.cLoc - self.x
        return
    def sample(self,sample):
        sample = np.array([[sample]])
        x = np.array([[self.x],[self.t]])
        x.shape=(2,1)
        sample.shape=(1,1)
        if self.world.X == None:
            self.world.infer(x,sample)
        else:
            self.world.infer_iter(x,sample)
        #self.world.optimise_hyper()
        self.find_cost()
        return
    def move(self):
        self.x = self.cLoc

        return
    def incomplete(self):
        if self.d == 0:
            return False
        else:
            return True

    def find_cost(self):
        Kdd = self.world.K
        fd = self.world.Y
        mult = np.linalg.solve(Kdd,fd)
        Xd = self.world.X
        self.cost = lambda c,t: 0 if (np.abs(c) >= 10) else - np.dot(np.exp(-np.dot(Xd - np.array([[c],[t]]).T,
            Xd - np.array([[c],[t]]))).T,mult) - self.world.get_Ys2(c)
        return
