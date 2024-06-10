import numpy as np
import fields

class SchwingerDyson:

    def __init__(self, beta, q, J, m, discretization, error_threshold, weight=0.05, max_iter=1000):

        self.q = np.double(q)
        self._beta = np.double(beta)
        self.J = np.double(J)
        self.Jsqr = np.double(J**2)
        self.m = np.double(m)
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
    def __get_Sigma(self):
        
        #denominator
        Gdij = fields.read_G_from_Ghat(self.Ghatd, int(self.discretization/2))

        brace = self.m/4*(-Gdij['G11'] + Gdij['G22'] - Gdij['G12'] + Gdij['G21']) + (1-self.m)*self.G33d
        Sigma_d11 = -self.normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_d_dict = {'G11': -Sigma_d11*2, 'G22': Sigma_d11*2, 'G12': Sigma_d11*2, 'G21': -Sigma_d11*2} #Sigma12 must be mapped with the G21 map and vice versa
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        self.Sigma33d = Sigma_d11 #sign flipped
        #print(max(np.amax(Sigma_d11),abs(np.amin(Sigma_d11))))

        self.Sigmahatd = fields.create_Sigma_hat(Sigma_d_dict,int(self.discretization/2))
       
        #numerator
        Gnij = fields.read_G_from_Ghat(self.Ghatn, int(self.discretization/2))

        brace = self.m/4*(-Gnij['G11'] + Gnij['G22'] -Gnij['G12'] + Gnij['G21']) + (1-self.m)*self.G33n
        Sigma_n11 = -self.normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_n_dict = {'G11': -Sigma_n11*2, 'G22': Sigma_n11*2, 'G12': Sigma_n11*2, 'G21': -Sigma_n11*2}
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        self.Sigma33n = Sigma_n11 #sign flipped

        self.Sigmahatn = fields.create_Sigma_hat(Sigma_n_dict,int(self.discretization/2))

    def __get_Sigma2(self, G33, Ghat):

        Gdij = fields.read_G_from_Ghat(Ghat, int(self.discretization/2))

        brace = self.m/4*(-Gdij['G11'] + Gdij['G22'] - Gdij['G12'] + Gdij['G21']) + (1-self.m)*G33
        Sigma33 = -self.normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_dict = {'G11': -Sigma33*2, 'G22': Sigma33*2, 'G12': Sigma33*2, 'G21': -Sigma33*2} #Sigma12 must be mapped with the G21 map and vice versa
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        return Sigma33, fields.create_Sigma_hat(Sigma_dict,int(self.discretization/2))
    
    #use first S.-D. equation to calculate Ghat and G33
    def __get_G(self):
    
        self.Ghatd = np.linalg.inv(self.Ghat_d_free_inverse.astype(np.double) - self.Sigmahatd.astype(np.double))
        self.Ghatn = np.linalg.inv(self.Ghat_n_free_inverse.astype(np.double) - self.Sigmahatn.astype(np.double))

        self.G33d = np.linalg.inv(self.G33_d_free_inverse.astype(np.double) - self.Sigma33d.astype(np.double)) 
        self.G33n = np.linalg.inv(self.G33_n_free_inverse.astype(np.double) - self.Sigma33n.astype(np.double)) 

    def __get_G2(self, Sigma33, Sigmahat, G33free_inv, Ghatfree_inv):

        Ghat = np.linalg.inv(Ghatfree_inv.astype(np.double) - Sigmahat.astype(np.double))
        G33 = np.linalg.inv(G33free_inv.astype(np.double) - Sigma33.astype(np.double))

        return G33, Ghat
  
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

        error = np.double(0)

        matrices = [(self.Ghatd - self.Ghatd_old)/(self.discretization*4), (self.G33d-self.G33d_old)/(self.discretization*2), (self.Ghatn - self.Ghatn_old)/(self.discretization*4), (self.G33n - self.G33n_old)/(self.discretization*2)]
        
        error = [np.abs(np.trace(matrix@matrix)) for matrix in matrices]

        return np.array([np.maximum(error[0],error[1]), np.max(error[2]+error[3])])
    
    def __get_error2(self, G33, Ghat, G33_old, Ghat_old):

        matrices = [(Ghat - Ghat_old)/(self.discretization*4), (G33-G33_old)/(self.discretization*2)]
        error = [np.abs(np.trace(matrix@matrix)) for matrix in matrices]
        return np.maximum(error[0], error[1])

    #iteratively solve the Schinger-Dyson equations
    def solve(self):

        weight = np.zeros(2, dtype=np.double)
        weight[:] = self.initial_weight

        self.__get_Sigma()
        self.__swap()
        self.__get_G()

        old_error = self.__get_error()

        self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
        self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d
        self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
        self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

        i = 1

        while(True):

            self.didconverge = (old_error <= self.error_threshold)
            if (all(self.didconverge)):
                break

            self.__get_Sigma()
            self.__swap()
            self.__get_G()

            error = self.__get_error()

            print(str(i)+ ". " +str(error))

            for j in range(len(error)):
                if error[j] > old_error[j]:
                    
                    weight[j] = weight[j]/2

            if (all(np.abs(old_error-error) < 1e-10*self.error_threshold)):
                print(np.abs(old_error-error))
                print("Error changerate below threshold. Resetting weight.\n")
                weight[:] = self.initial_weight

            old_error = error

            self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
            self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d
            self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
            self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

            i += 1

            if (i > self.max_iter):
                print("Warning: max. number of iterations reached.\n")
                break

        self.iter_count = i

    def solve2(self):
        
        error = np.zeros(2, dtype=np.double)
        old_error = np.zeros(2, dtype=np.double)
        weight = np.zeros(2, dtype=np.double)
        weight[:] = self.initial_weight

        #numerator
        self.Sigma33n, self.Sigmahatn = self.__get_Sigma2(self.G33n,self.Ghatn)
        self.G33n_old = self.G33n
        self.Ghatn_old = self.Ghatn
        self.G33n, self.Ghatn = self.__get_G2(self.Sigma33n, self.Sigmahatn, self.G33_n_free_inverse, self.Ghat_n_free_inverse)
        old_error[1] = self.__get_error2(self.G33n,self.Ghatn,self.G33n_old,self.Ghatn_old)
        self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
        self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

        #denominator
        self.Sigma33d, self.Sigmahatd = self.__get_Sigma2(self.G33d,self.Ghatd)
        self.G33d_old = self.G33d
        self.Ghatd_old = self.Ghatd
        self.G33d, self.Ghatd = self.__get_G2(self.Sigma33d, self.Sigmahatd, self.G33_d_free_inverse, self.Ghat_d_free_inverse)
        old_error[0] = self.__get_error2(self.G33d,self.Ghatd,self.G33d_old,self.Ghatd_old)
        self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
        self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d

        i = 1

        while(True):
            
            if (i >= self.max_iter):
                print("Warning: max. number of iterations reached.\n")
                break

            self.didconverge = (old_error <= self.error_threshold)
            if (all(self.didconverge)):
                break

            i += 1

            if (not self.didconverge[1]):
                self.Sigma33n, self.Sigmahatn = self.__get_Sigma2(self.G33n,self.Ghatn)
                self.G33n_old = self.G33n
                self.Ghatn_old = self.Ghatn
                self.G33n, self.Ghatn = self.__get_G2(self.Sigma33n, self.Sigmahatn, self.G33_n_free_inverse, self.Ghat_n_free_inverse)
                error[1] = self.__get_error2(self.G33n,self.Ghatn,self.G33n_old,self.Ghatn_old)
                if (error[1] > old_error[1]):
                    weight[1] = weight[1]/2
                if (abs(old_error[1] - error[1]) < 1e-10*self.error_threshold ):
                    print("Reset weight for numerator.\n")
                    weight[1] = self.initial_weight
                self.Ghatn = (1-weight[1])*self.Ghatn_old + weight[1]*self.Ghatn
                self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

            if (not self.didconverge[0]):
                self.Sigma33d, self.Sigmahatd = self.__get_Sigma2(self.G33d,self.Ghatd)
                self.G33d_old = self.G33d
                self.Ghatd_old = self.Ghatd
                self.G33d, self.Ghatd = self.__get_G2(self.Sigma33d, self.Sigmahatd, self.G33_d_free_inverse, self.Ghat_d_free_inverse)
                error[0] = self.__get_error2(self.G33d,self.Ghatd,self.G33d_old,self.Ghatd_old)
                if (error[0] > old_error[0]):
                    weight[0] = weight[0]/2
                if (abs(old_error[0] - error[0]) < 1e-10*self.error_threshold ):
                    print("Reset weight for denominator.\n")
                    weight[0] = self.initial_weight
                self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
                self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d


            print(str(i)+ ". " +str(error))
            old_error[:] = error[:]










