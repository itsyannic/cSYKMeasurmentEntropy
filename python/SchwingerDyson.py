import numpy as np
import fields

class SchwingerDyson:

    def __init__(self, beta, q, J, m, discretization, error_threshold, weight=0.05, max_iter=1000, silent=False):

        self.q = np.double(q)
        self._beta = np.double(beta)
        self.J = np.double(J)
        self.Jsqr = np.double(J**2)
        self.m = np.double(m)
        self.discretization = discretization
        self.max_iter = max_iter
        self.silent = silent

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

        self.error_threshold = np.double(error_threshold)
        self.initial_weight = np.double(weight)
        self.normalization = np.power(self._beta/self.discretization,2)
        self.iter_count = 0
        self.didconverge = [False,False]

        self.init_matrices()

        self.Ghatdfree = self.Ghatd
        self.Ghatnfree = self.Ghatn

        self.G33dfree = self.G33d
        self.G33nfree = self.G33n

        self.Ghat_d_free_inverse = np.linalg.inv(self.Ghatd)
        self.Ghat_n_free_inverse = np.linalg.inv(self.Ghatn)

        self.G33_d_free_inverse = np.linalg.inv(self.G33d)
        self.G33_n_free_inverse = np.linalg.inv(self.G33n)

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, input):

        self._beta = input
        self.normalization = np.power(self._beta/self.discretization,2)


    def init_matrices(self):

        for i in range(2*self.discretization):
            for j in range(2*self.discretization):
                self.G33n[i,j] = 0.5*np.sign(i-j, dtype=np.double)

                if ((i<self.discretization and j<self.discretization) or (i >= self.discretization and j>=self.discretization )):
                    self.G33d[i,j] = 0.5*np.sign(i-j, dtype=np.double)
                else:
                    self.G33d[i,j] = 0

        for i in range(4*self.discretization):
            for j in range(4*self.discretization):

                if ( ((self.discretization<=i<3*self.discretization) and (self.discretization<=j<3*self.discretization)) or (( i< self.discretization or 3*self.discretization<=i<4*self.discretization) and ( j < self.discretization or 3*self.discretization<=j<4*self.discretization))):
                    self.Ghatn[i,j] = 0.5*np.sign(i-j, dtype=np.double)
                else:
                    self.Ghatn[i,j] = 0

                if ((i<2*self.discretization and j<2*self.discretization) or (i >= 2*self.discretization and j>=2*self.discretization )):
                    self.Ghatd[i,j] = 0.5*np.sign(i-j, dtype=np.double)
                else:
                    self.Ghatd[i,j] = 0

    def reset(self):

        self.Ghatd = self.Ghatdfree
        self.Ghatn = self.Ghatnfree

        self.G33d = self.G33dfree
        self.G33n = self.G33nfree

    #Calculate Gijs from Ghat, use second S.-D. equation to calculate Sigma_ijs and then calculate Sigma_hat
    
    def __get_Sigma(self, G33, Ghat):

        Gdij = fields.read_G_from_Ghat(Ghat, int(self.discretization/2))

        brace = self.m/4*(-Gdij['G11'] + Gdij['G22'] - Gdij['G12'] + Gdij['G21']) + (1-self.m)*G33
        Sigma33 = -self.normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_dict = {'G11': -Sigma33*2, 'G22': Sigma33*2, 'G12': Sigma33*2, 'G21': -Sigma33*2} #Sigma12 must be mapped with the G21 map and vice versa
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        return Sigma33, fields.create_Sigma_hat(Sigma_dict,int(self.discretization/2))
    
    #use first S.-D. equation to calculate Ghat and G33

    def __get_G(self, Sigma33, Sigmahat, G33free_inv, Ghatfree_inv):

        Ghat = np.linalg.inv(Ghatfree_inv.astype(np.double) - Sigmahat.astype(np.double))
        G33 = np.linalg.inv(G33free_inv.astype(np.double) - Sigma33.astype(np.double))

        return G33, Ghat
    
    def __get_error(self, G33, Ghat, G33_old, Ghat_old):

        matrices = [(Ghat - Ghat_old), (G33-G33_old)]
        error = [np.abs(np.trace(matrix@matrix)) for matrix in matrices]
        error[0] = error[0]/(4*self.discretization**2)
        error[1] = error[1]/(2*self.discretization**2)
        return error[0] + error[1]

    def solve(self):
        
        error = np.zeros(2, dtype=np.double)
        old_error = np.zeros(2, dtype=np.double)
        weight = np.zeros(2, dtype=np.double)
        weight[:] = self.initial_weight

        #numerator
        self.Sigma33n, self.Sigmahatn = self.__get_Sigma(self.G33n,self.Ghatn)
        self.G33n_old = self.G33n
        self.Ghatn_old = self.Ghatn
        self.G33n, self.Ghatn = self.__get_G(self.Sigma33n, self.Sigmahatn, self.G33_n_free_inverse, self.Ghat_n_free_inverse)
        old_error[1] = self.__get_error(self.G33n,self.Ghatn,self.G33n_old,self.Ghatn_old)
        self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
        self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

        #denominator
        self.Sigma33d, self.Sigmahatd = self.__get_Sigma(self.G33d,self.Ghatd)
        self.G33d_old = self.G33d
        self.Ghatd_old = self.Ghatd
        self.G33d, self.Ghatd = self.__get_G(self.Sigma33d, self.Sigmahatd, self.G33_d_free_inverse, self.Ghat_d_free_inverse)
        old_error[0] = self.__get_error(self.G33d,self.Ghatd,self.G33d_old,self.Ghatd_old)
        self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
        self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d

        i = 1

        while(True):
            
            if (i >= self.max_iter):
                if (not self.silent):
                    print("Warning: max. number of iterations reached.\n")
                break

            self.didconverge = (old_error <= self.error_threshold)
            if (all(self.didconverge)):
                break

            i += 1

            if (not self.didconverge[1]):
                self.Sigma33n, self.Sigmahatn = self.__get_Sigma(self.G33n,self.Ghatn)
                self.G33n_old = self.G33n
                self.Ghatn_old = self.Ghatn
                self.G33n, self.Ghatn = self.__get_G(self.Sigma33n, self.Sigmahatn, self.G33_n_free_inverse, self.Ghat_n_free_inverse)
                error[1] = self.__get_error(self.G33n,self.Ghatn,self.G33n_old,self.Ghatn_old)
                if (error[1] > old_error[1]):
                    weight[1] = weight[1]/2
                if (abs(old_error[1] - error[1]) < 1e-6*error[1]):
                    if (not self.silent):
                        print("Reset weight for numerator.\n")
                    weight[1] = self.initial_weight
                self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
                self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

            if (not self.didconverge[0]):
                self.Sigma33d, self.Sigmahatd = self.__get_Sigma(self.G33d,self.Ghatd)
                self.G33d_old = self.G33d
                self.Ghatd_old = self.Ghatd
                self.G33d, self.Ghatd = self.__get_G(self.Sigma33d, self.Sigmahatd, self.G33_d_free_inverse, self.Ghat_d_free_inverse)
                error[0] = self.__get_error(self.G33d,self.Ghatd,self.G33d_old,self.Ghatd_old)
                if (error[0] > old_error[0]):
                    weight[0] = weight[0]/2
                if (abs(old_error[0] - error[0]) < 1e-6*error[0] ):
                    if (not self.silent):
                        print("Reset weight for denominator.\n")
                    weight[0] = self.initial_weight
                self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
                self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d

            if (not self.silent):
                print(str(i)+ ". " +str(error))
            old_error[:] = error[:]










