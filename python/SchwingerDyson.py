import numpy as np

class SchwingerDyson:

    def __init__(self, beta, q, J, discretization, steps=1000):

        self.q = q
        self.beta = beta
        self.J = J
        self.discretization = discretization
        self.steps = steps

        self.G33n = np.zeros((2*discretization, 2*discretization))
        self.G33d = np.zeros((2*discretization, 2*discretization))
        self.Ghatn = np.zeros((4*discretization, 4*discretization))
        self.Ghatd = np.zeros((4*discretization, 4*discretization))

        self.init_matrices()

    def init_matrices(self):

        for i in range(2*self.discretization):
            for j in range(2*self.discretization):
                self.G33n[i,j] = 0.5*np.sign(i-j)

                if ((i<self.discretization and j<self.discretization) or (i >= self.discretization and j>=self.discretization )):
                    self.G33d[i,j] = 0.5*np.sign(i-j)
                else:
                    self.G33d[i,j] = 0

        for i in range(4*self.discretization):
            for j in range(4*self.discretization):

                if ( ((self.discretization<=i<3*self.discretization) and (self.discretization<=j<3*self.discretization)) or (( i< self.discretization or 3*self.discretization<=i<4*self.discretization) and ( j< self.discretization or 3*self.discretization<=j<4*self.discretization))):
                    self.Ghatn[i,j] = 0.5*np.sign(i-j)
                else:
                    self.Ghatn[i,j] = 0

                if ((i<2*self.discretization and j<2*self.discretization) or (i >= 2*self.discretization and j>=2*self.discretization )):
                    self.Ghatd[i,j] = 0.5*np.sign(i-j)
                else:
                    self.Ghatd[i,j] = 0


    def solve(self):

        return

