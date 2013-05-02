import numpy as np

class Reward:
    def __init__(self,centers,magnitudes,speeds,frequencies):
        self.centers = centers
        self.magnitudes = magnitudes
        self.speeds = speeds
        self.frequencies = frequencies
       
    def calc(self,x,t):
        r = 0
        for c,m,s,f in zip(self.centers,self.magnitudes,self.speeds,self.frequencies):
            r+= m*np.sin(f*t)*np.exp(-0.5 * ((x.T-c).dot((x.T-c).T)))
        return r
