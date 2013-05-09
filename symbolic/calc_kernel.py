import numpy as np
import sympy as sp
from sympy import Matrix
from sympy import MatrixSymbol

#bleh
Z_x = MatrixSymbol('Z_x',3,1)
#Current location
Z_loc = MatrixSymbol('Z_loc',3,1)

#Control location
Z_con = MatrixSymbol('Z_con',3,1)

#Data Location
Z_dat = MatrixSymbol('A',3,1)

#GP Mean function (SqExp)
ell = sp.Symbol('ell')
S_mean = MatrixSymbol('S_mean',3,3)
mean = sp.exp(- (Z_dat - Z_x).T*S_mean.I*(Z_dat - Z_x))

#Agent control
prec = sp.Symbol('prec')
sig2 = MatrixSymbol('sig2',3,3)
agent =(1/sp.sqrt(2*np.pi)) * sp.exp(- (Z_loc - Z_con).T * sig2.I * (Z_loc - Z_con))

f = mean * agent
print(sp.simplify(f))
