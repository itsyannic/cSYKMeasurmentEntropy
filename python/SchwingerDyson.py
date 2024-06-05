import numpy as np
import fields

class SchwingerDyson:

    def __init__(self, beta, q, J, m, discretization, error_threshold, weight=0.05, max_iter=1000):

        self.q = np.double(q)
        self.beta = np.double(beta)
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
        self.__normalization = np.power(self.beta/self.discretization,2)
        self.iter_count = 0

        self.init_matrices()

        self.Ghat_d_free_inverse = np.linalg.inv(self.Ghatd)
        self.Ghat_n_free_inverse = np.linalg.inv(self.Ghatn)

        self.G33_d_free_inverse = np.linalg.inv(self.G33d)
        self.G33_n_free_inverse = np.linalg.inv(self.G33n)

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

    #Calculate Gijs from Ghat, use second S.-D. equation to calculate Sigma_ijs and then calculate Sigma_hat
    def __get_Sigma(self):
        
        #denominator
        Gdij = fields.read_G_from_Ghat(self.Ghatd, int(self.discretization/2))

        brace = self.m/4*(-Gdij['G11'] + Gdij['G22'] - Gdij['G12'] + Gdij['G21']) + (1-self.m)*self.G33d
        Sigma_d11 = -self.__normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_d_dict = {'G11': Sigma_d11*2, 'G22': -Sigma_d11*2, 'G12': -Sigma_d11*2, 'G21': Sigma_d11*2} #Sigma12 must be mapped with the G21 map and vice versa
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        self.Sigma33d = Sigma_d11 #sign flipped
        #print(max(np.amax(Sigma_d11),abs(np.amin(Sigma_d11))))

        self.Sigmahatd = fields.create_Sigma_hat(Sigma_d_dict,int(self.discretization/2))
       
        #numerator
        Gnij = fields.read_G_from_Ghat(self.Ghatn, int(self.discretization/2))

        brace = self.m/4*(-Gnij['G11'] + Gnij['G22'] -Gnij['G12'] + Gnij['G21']) + (1-self.m)*self.G33n
        Sigma_n11 = -self.__normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma_n_dict = {'G11': Sigma_n11*2, 'G22': -Sigma_n11*2, 'G12': -Sigma_n11*2, 'G21': Sigma_n11*2}
        #the factor of two is ther to account for the fact that the Ghat matrix has a 1/2 in front while Sigmahat does not
        #update Sigma matrices
        self.Sigma33n = Sigma_n11 #sign flipped

        self.Sigmahatn = fields.create_Sigma_hat(Sigma_n_dict,int(self.discretization/2))
    
    #use first S.-D. equation to calculate Ghat and G33
    def __get_G(self):
    
        self.Ghatd = np.linalg.inv(self.Ghat_d_free_inverse.astype(np.double) - self.Sigmahatd.astype(np.double))
        self.Ghatn = np.linalg.inv(self.Ghat_n_free_inverse.astype(np.double) - self.Sigmahatn.astype(np.double))

        self.G33d = np.linalg.inv(-self.G33_d_free_inverse.astype(np.double) - self.Sigma33d.astype(np.double)) #sign flipped
        self.G33n = np.linalg.inv(-self.G33_n_free_inverse.astype(np.double) - self.Sigma33n.astype(np.double)) #sign flipped
    
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

        return error

    #iteratively solve the Schinger-Dyson equations
    def solve(self):

        weight = np.zeros(4, dtype=np.double)
        weight[:] = self.initial_weight

        self.__get_Sigma()
        self.__swap()
        self.__get_G()

        old_error = self.__get_error()

        self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
        self.G33d = (1-weight[1])*self.G33d_old + weight[1]*self.G33d
        self.Ghatn = (1-weight[2])*self.Ghatn_old + weight[2]*self.Ghatn
        self.G33n = (1-weight[3])*self.G33n_old + weight[3]*self.G33n

        i = 1

        while(np.max(old_error) >= self.error_threshold):

            self.__get_Sigma()
            self.__swap()
            self.__get_G()

            error = self.__get_error()

            print(error)

            for i in range(len(error)):
                if error[i] > old_error[i]:

                    weight[i] = weight[i]/2
                    #print(weight)
                    #print("Weight updated at iteration step n = " + str(i) + ": x = "+ str(weight) + "\n")

            old_error = error

            self.Ghatd = (1-weight[0])*self.Ghatd_old + weight[0]*self.Ghatd
            self.G33d = (1-weight[1])*self.G33d_old + weight[1]*self.G33d
            self.Ghatn = (1-weight[2])*self.Ghatn_old + weight[2]*self.Ghatn
            self.G33n = (1-weight[3])*self.G33n_old + weight[3]*self.G33n

            i += 1

            if (i > self.max_iter):
                print("Warning: max. number of iterations reached.\n")
                break

        self.iter_count = i




