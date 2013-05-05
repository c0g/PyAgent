import numpy as np

class Reward:
    def __init__(self,centers,magnitudes,speeds,frequencies):
        self.centers = centers
        self.magnitudes = magnitudes
        self.speeds = speeds
        self.frequencies = frequencies
       
    def calc(self,Z):
        return -np.linalg.norm(Z[:-1]) 
