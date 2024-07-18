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

        brace = self.m*(Gdij) + (1-self.m)*self.G33d
        Sigma_d33 = -self.__normalization*self.Jsqr*np.power(brace,self.q/2)*np.transpose(np.power(brace,self.q/2-1))
        Sigma_d11 = (Sigma_d33+np.transpose(Sigma_d33))/2
        Sigma_d22 = (Sigma_d33-np.transpose(Sigma_d33))/2
        Sigma_d_dict = {'Gm': Sigma_d11, 'Gmtilde': Sigma_d22}


        self.Sigmahatd = fields.create_Sigma_hat(Sigma_d_dict,int(self.discretization/2))
       
        #numerator
        Gnij = fields.read_G_from_Ghat(self.Ghatn, int(self.discretization/2))

        brace = self.m*(Gnij) + (1-self.m)*self.G33d
        Sigma_n33 = -self.__normalization*self.Jsqr*np.power(brace,self.q/2)*np.transpose(np.power(brace,self.q/2-1))
        Sigma_n11 = (Sigma_n33+np.transpose(Sigma_n33))/2
        Sigma_n22 = (Sigma_n33-np.transpose(Sigma_n33))/2
        Sigma_n_dict = {'Gm': Sigma_n11, 'Gmtilde': Sigma_n22}

        self.Sigmahatn = fields.create_Sigma_hat(Sigma_n_dict,int(self.discretization/2))
    
    #use first S.-D. equation to calculate Ghat and G33
    def __get_G(self, weight):
    
        self.Ghatd = (1-weight)*self.Ghatd_old + weight*np.linalg.inv(self.Ghat_d_free_inverse.astype(np.double) - self.Sigmahatd.astype(np.complex64))
        self.Ghatn = (1-weight)*self.Ghatn_old + weight*np.linalg.inv(self.Ghat_n_free_inverse.astype(np.double) - self.Sigmahatn.astype(np.complex64))

        self.G33d = (1-weight)*self.G33d_old + weight*np.linalg.inv(self.G33_d_free_inverse.astype(np.double) - self.Sigma33d.astype(np.complex64))
        self.G33n = (1-weight)*self.G33n_old + weight*np.linalg.inv(self.G33_n_free_inverse.astype(np.double) - self.Sigma33n.astype(np.complex64))
    
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

        matrices = [(self.Ghatd - self.Ghatd_old)/(self.discretization*4), (self.Ghatn - self.Ghatn_old)/(self.discretization*4), 
                    (self.G33d-self.G33d_old)/(self.discretization*2), (self.G33n - self.G33n_old)/(self.discretization*2)]
        
        error = [np.abs(np.trace(matrix@matrix)) for matrix in matrices]

        return np.max(error)

    #iteratively solve the Schinger-Dyson equations
    def solve(self):

        weight = self.initial_weight

        self.__get_Sigma()
        self.__swap()
        self.__get_G(weight)

        old_error = self.__get_error()

        i = 1

        while(old_error >= weight**2*self.error_threshold):

            self.__get_Sigma()
            self.__swap()
            self.__get_G(weight)

            error = self.__get_error()
            #print(error/weight)

            if error > old_error:
                weight = weight/2
                #print("Weight updated at iteration step n = " + str(i) + ": x = "+ str(weight) + "\n")

            old_error = error
            print(error)

            i += 1

            if (i > self.max_iter):
                print("Warning: max. number of iterations reached.\n")
                break

        self.iter_count = i




