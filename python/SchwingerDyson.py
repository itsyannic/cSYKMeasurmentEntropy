import numpy as np
import fields

class SchwingerDyson:

    def __init__(self, beta, q, J, m, discretization, error_threshold, weight=0.05, max_iter=1000):

        self.q = q
        self.beta = beta
        self.J = J
        self.Jsqr = J**2
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

        self.Ghat_d_free_inverse = np.linalg.inv(self.Ghatd)
        self.Ghat_n_free_inverse = np.linalg.inv(self.Ghatn)

        self.G33_d_free_inverse = np.linalg.inv(self.G33d)
        self.G33_n_free_inverse = np.linalg.inv(self.G33n)

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
    def __get_Sigma(self):

        Gdij = fields.read_G_from_Ghat(self.Ghatd, int(self.discretization/2))
        Gnij = fields.read_G_from_Ghat(self.Ghatn, int(self.discretization/2))
        
        #denominator
        brace = -self.m/2*Gdij['G11'] + self.m/2*Gdij['G22'] - self.m/2*Gdij['G12'] + self.m/2*Gdij['G21'] + (1-self.m)*self.G33d
        Sigma_d11 = -2*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_d_dict = {'G11': Sigma_d11, 'G22': -Sigma_d11, 'G12': -Sigma_d11, 'G21': Sigma_d11} #Sigma12 must be mapped with the G21 map and vice versa
        #update Sigma matrices
        self.Sigma33d = -Sigma_d11

        self.Sigmahatd = fields.create_Sigma_hat(Sigma_d_dict,int(self.discretization/2))
       
        #numerator
        brace = -self.m/2*Gnij['G11'] + self.m/2*Gnij['G22'] - self.m/2*Gnij['G12'] + self.m/2*Gnij['G21'] + (1-self.m)*self.G33n
        Sigma_n11 = -2*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        Sigma_n_dict = {'G11': Sigma_n11, 'G22': -Sigma_n11, 'G12': -Sigma_n11, 'G21': Sigma_n11}
        #update Sigma matrices
        self.Sigma33n = -Sigma_n11

        self.Sigmahatn = fields.create_Sigma_hat(Sigma_n_dict,int(self.discretization/2))
    
    #use first S.-D. equation to calculate Ghat and G33
    def __get_G(self, weight):
    
        self.Ghatd = (1-weight)*self.Ghatd_old+weight*np.linalg.inv(self.Ghat_d_free_inverse.astype(np.double) - self.Sigmahatd.astype(np.double))
        self.Ghatn = (1-weight)*self.Ghatn_old+weight*np.linalg.inv(self.Ghat_n_free_inverse.astype(np.double) - self.Sigmahatn.astype(np.double))

        self.G33d = (1-weight)*self.G33d_old+weight*np.linalg.inv(self.G33_d_free_inverse.astype(np.double) - self.Sigma33d.astype(np.double))
        self.G33n = (1-weight)*self.G33n_old+weight*np.linalg.inv(self.G33_n_free_inverse.astype(np.double) - self.Sigma33n.astype(np.double))
    
    #swap the labels for the old and the current matrix
    def __swap(self):
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

    #returns the largest error found    
    def __get_error(self):

        error = 0

        matrices = [self.Ghatd - self.Ghatd_old, self.Ghatn - self.Ghatn_old, self.G33d-self.G33d_old, self.G33n - self.G33n_old]
        for matrix in matrices:
            e = abs(np.trace(matrix@matrix))
            if (e > error):
                error = e

        return error

    #iteratively solve the Schinger-Dyson equations
    def solve(self):

        weight = self.initial_weight

        self.__get_Sigma()
        self.__swap()
        self.__get_G(weight)

        old_error = self.__get_error()

        i = 1

        while(old_error >= self.error_threshold):

            self.__get_Sigma()
            self.__swap()
            self.__get_G(weight)

            error = self.__get_error()

            if error > old_error:
                weight = weight/2

            old_error = error

            i += 1

            if (i > self.max_iter):
                break

        self.iter_count = i




