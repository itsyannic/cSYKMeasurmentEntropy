import numpy as np
import fields

class SchwingerDyson:

    def __init__(self, beta, q, J, m, discretization, error_threshold, weight=0.05, max_iter=1000):

        self.q = q
        self.beta = beta
        self.J = J
        self.m = m
        self.discretization = discretization
        self.max_iter = max_iter

        self.G33n = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G33d = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Ghatn = np.zeros((4*discretization, 4*discretization), dtype=np.double)
        self.Ghatd = np.zeros((4*discretization, 4*discretization), dtype=np.double)

        self.G33n_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G33d_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Ghatn_old = np.zeros((4*discretization, 4*discretization), dtype=np.double)
        self.Ghatd_old = np.zeros((4*discretization, 4*discretization), dtype=np.double)

        self.Sigmahatn = np.zeros((4*discretization, 4*discretization), dtype=np.double)
        self.Sigmahatd = np.zeros((4*discretization, 4*discretization), dtype=np.double)
        self.Sigma33n = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Sigma33d = np.zeros((2*discretization, 2*discretization), dtype=np.double)

        self.error_threshold = error_threshold
        self.initial_weight = weight

        self.iter_count = 0

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

    #Calculate Gijs from Ghat, use second S.-D. equation to calculate Sigma_ijs and then calculate Sigma_hat
    def get_Sigma(self):

        Gdij = fields.read_G_from_Ghat(self.Ghatd)
        Gnij = fields.read_G_from_Ghat(self.Ghatn)

        return
    
    #use first S.-D. equation to calculate Ghat and G33
    def get_G(self, weight):

        return
    
    #swap the labels for the old and the current matrix
    def swap(self):
        aux = self.Ghatd_old
        self.Ghatd_old = self.Ghatd
        self.Ghatd = aux

        aux = self.Ghatn_old
        self.Ghatn_old = self.Ghatn
        self.Ghatn = aux
        
        aux = self.G33d_old
        self.G33d_old = self.G33d
        self.G33d = aux

        aux = self.G33n_old
        self.G33n_old = self.G33n
        self.G33n = aux
        
    def get_error(self):

        return

    #iteratively solve the Schinger-Dyson equations
    def solve(self):

        weight = self.initial_weight

        self.get_Sigma()
        self.swap()
        self.get_G(weight)

        old_error = self.get_error()

        i = 1

        while(old_error >= self.error_threshold):

            self.get_Sigma()
            self.swap()
            self.get_G(weight)

            error = self.get_error()

            if error > old_error:
                weight = weight/2

            old_error = error

            i += 1

            if (i > self.max_iter):
                break

        self.iter_count = i




