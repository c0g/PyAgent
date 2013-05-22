import numpy as np
import scipy.optimize as opt
from solve_for_c import solve_for_c


def solve_chol(A, b):
    """Attempt to solve using cholesky decomp of A. If that fails, use the other one"""
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, b))
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, b)


class Agent:

    def __init__(self, gp, reward, sig, start_z):
        self.Sigma_c = np.eye(start_z.shape[1]) * sig**2
        self.gp = gp
        self.reward = reward
        self.z = start_z
        self.z_next = start_z

    def act(self):
        self.z = self.z_next

    def observe(self):
        r = np.array([self.reward.calc(self.z)])
        self.gp.observe(np.array([self.z.flatten()]), r)

    def decide(self):
        self.z_next = self.optimise_mean()

    def mean(self, control):
        """Reset the last element of control to be current time + 1"""
        control[-1] = self.gp.Z[-1].flatten()[-1] + 1
        Sigma = self.gp.cov.Sigma + self.Sigma_c
        Kddifd = solve_chol(self.gp.cov.K(self.gp.hyp, self.gp.Z, self.gp.Z)[0], self.gp.F)
        """Define a function that encloses our values for control, Sigma and Kddifd and returns the normal"""

        def normal(zd):
            return (1/np.sqrt(np.linalg.det(2*np.pi*Sigma))) * np.exp(- (zd - control).T.dot(solve_chol(2*Sigma, (zd-control))))

        def dnormal(zd):
            return -solve_chol(2*Sigma, (control-zd)) * normal(zd) * 1e10
        f = 0
        df = 0
        for zd, w in zip(self.gp.Z, Kddifd):
            f += w*normal(zd)
            df += w*dnormal(zd)
        df[-1] = 0
        return(f, df.flatten())

    def optimise_mean(self):
        """Initialise control to be the current location, at the next time step"""
        control = self.gp.Z[-1].flatten()
        control[-1] += 1

        """Define the constraints to ensure the agent moves 1 unit of space"""
        def con(control, gp):
            x = gp.Z[-1].flatten()[:-1]
            dist = np.linalg.norm(control[:-1] - x)
            return 0.1 - dist**2

        def dcon(control, gp):
            x = gp.Z[-1].flatten()[:-1]
            return(- 2 * (control[:-1] - x))

        cons = {'type': 'eq',
                'fun': con,
                'dfun': dcon,
                'args': [self.gp]}
        ret = opt.minimize(self.mean, control, jac=True, method='SLSQP', constraints=cons)
        return(ret.x)