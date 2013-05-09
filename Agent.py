import numpy as np
import scipy as sp
from solve_for_c import solve_for_c

class Agent:
    def __init__(self,Z,world,worldModel,utility):
        #Current location
        dim = np.shape(Z)[1]
        self.Z = Z #current position
        self.cLoc = Z #control location
        self.world = world #function expresssing the true reward function
        self.worldModel = worldModel #this is our GP
        self.moveSig = 0.01 #our uncertainty about future location, given a control signal
        self.Sigmainv = worldModel.hyp[0]*np.eye(dim) + np.diag([self.moveSig] * (dim-1) + [0]) 
        self.movU = 0
        self.utility = utility #Callable taking Ymu and Ys2
        return
    def plan(self):
        # Agent finds optimal move, and stores the expected utility of that
        cLoc = solve_for_c(self.Sigmainv,self.worldModel.K,self.worldModel.F,self.worldModel.Z,self.Z,self.utility)
        self.cLoc = cLoc
        self.worldModel.predict(self.cLoc)
        self.movU = self.utility(self.worldModel.Ymu,self.worldModel.Ys2,self.cLoc)
        self.worldModel.predict(self.Z) 
        self.samU = self.utility(self.worldModel.Ymu,self.worldModel.Ys2,self.Z)
        return
    def execute(self):
        self.move()
        self.sample()

    def move(self):
        print("Moving to: " +str(self.cLoc) + "\n")
        self.Z = self.cLoc
        return
    def sample(self):
        print("Sampling at " + str(self.Z) + "\n")
        self.Z[0][-1] += 1
        sample = self.world.draw(self.Z)
        self.world.infer(self.Z,sample)
        self.worldModel.infer(self.Z,sample)
        return
