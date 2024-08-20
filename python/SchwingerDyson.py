import numpy as np

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
        self.G11n = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G11d = np.zeros((2*discretization, 2*discretization), dtype=np.double)

        self.G33n_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G33d_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G11n_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.G11d_old = np.zeros((2*discretization, 2*discretization), dtype=np.double)

        self.Sigma11n = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Sigma11d = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Sigma33n = np.zeros((2*discretization, 2*discretization), dtype=np.double)
        self.Sigma33d = np.zeros((2*discretization, 2*discretization), dtype=np.double)

        self.error_threshold = np.double(error_threshold)
        self.initial_weight = np.double(weight)
        self.normalization = np.power(self._beta/self.discretization,2)
        self.iter_count = 0
        self.didconverge = [False,False]

        self.init_matrices()

        self.G11dfree = self.G11d
        self.G11nfree = self.G11n

        self.G33dfree = self.G33d
        self.G33nfree = self.G33n

        self.G11_d_free_inverse = np.linalg.solve(self.G11d, np.identity(2*self.discretization,dtype=np.double))
        self.G11_n_free_inverse = np.linalg.solve(self.G11n, np.identity(2*self.discretization,dtype=np.double))

        self.G33_d_free_inverse = np.linalg.solve(self.G33d, np.identity(2*self.discretization,dtype=np.double))
        self.G33_n_free_inverse = np.linalg.solve(self.G33n, np.identity(2*self.discretization,dtype=np.double))

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

        length = self.discretization
        string = np.concatenate( (np.full(length, 1, dtype=np.double),
                                  np.full(length, 0, dtype=np.double),
                                  np.full(length, -1, dtype=np.double), 
                                  np.full(int(length), 0, dtype=np.double)
                                  ) )
        M = np.array([string[2*length-i:(4*length-i)] for i in range(2*length)], dtype=np.double)

        self.G11n = M

        string = np.concatenate( (np.full(int(length/2), 1, dtype=np.double),
                                  np.full(int(length/2), 0, dtype=np.double),
                                  np.full(int(length/2), -1, dtype=np.double), 
                                  np.full(int(length/2), 0, dtype=np.double)
                                  ) )
        M = np.array([string[int(length)-i:(int(2*length)-i)] for i in range(length)], dtype=np.double)
        M = np.block([[M,np.zeros((length,length))], [np.zeros((length,length)),M]])

        self.G11d = M


    def reset(self):

        self.G11d = self.G11dfree
        self.G11n = self.G11nfree

        self.G33d = self.G33dfree
        self.G33n = self.G33nfree

    #Use second S.-D. equation to calculate Sigma_ijs
    
    def __get_Sigma(self, G33, G11):

        brace = self.m*G11 + (1-self.m)*G33
        Sigma33 = -self.normalization*self.Jsqr*np.multiply(np.power(brace,self.q/2), np.power(np.transpose(brace),self.q/2-1))
        Sigma11 = Sigma33
       
        return Sigma33, Sigma11
    
    #use first S.-D. equation to calculate G11 and G33

    def __get_G(self, Sigma33, Sigma11, G33free_inv, G11free_inv):

        G11 = np.linalg.solve(G11free_inv.astype(np.double) - Sigma11.astype(np.double), np.identity(2*self.discretization,dtype=np.double))
        G33 = np.linalg.solve(G33free_inv.astype(np.double) - Sigma33.astype(np.double), np.identity(2*self.discretization,dtype=np.double))

        return G33, G11
    
    def __get_error(self, G33, G11, G33_old, G11_old):

        matrices = [(G11 - G11_old), (G33-G33_old)]
        error = [np.abs(np.trace(matrix@matrix))*self.normalization for matrix in matrices]
        return error[0] + error[1]

    def solve(self):
        
        error = np.zeros(2, dtype=np.double)
        old_error = np.zeros(2, dtype=np.double)
        weight = np.zeros(2, dtype=np.double)
        weight[:] = self.initial_weight

        #numerator
        self.Sigma33n, self.Sigma11n = self.__get_Sigma(self.G33n,self.G11n)
        self.G33n_old = self.G33n
        self.G11n_old = self.G11n
        self.G33n, self.G11n = self.__get_G(self.Sigma33n, self.Sigma11n, self.G33_n_free_inverse, self.G11_n_free_inverse)
        old_error[1] = self.__get_error(self.G33n,self.G11n,self.G33n_old,self.G11n_old)
        self.G11n = (1-weight[1])*self.G11n_old + weight[1]*self.G11n
        self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

        #denominator
        self.Sigma33d, self.Sigma11d = self.__get_Sigma(self.G33d,self.G11d)
        self.G33d_old = self.G33d
        self.G11d_old = self.G11d
        self.G33d, self.G11d = self.__get_G(self.Sigma33d, self.Sigma11d, self.G33_d_free_inverse, self.G11_d_free_inverse)
        old_error[0] = self.__get_error(self.G33d,self.G11d,self.G33d_old,self.G11d_old)
        self.G11d = (1-weight[0])*self.G11d_old + weight[0]*self.G11d
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
                self.Sigma33n, self.Sigma11n = self.__get_Sigma(self.G33n,self.G11n)
                self.G33n_old = self.G33n
                self.G11n_old = self.G11n
                self.G33n, self.G11n = self.__get_G(self.Sigma33n, self.Sigma11n, self.G33_n_free_inverse, self.G11_n_free_inverse)
                error[1] = self.__get_error(self.G33n,self.G11n,self.G33n_old,self.G11n_old)
                if (error[1] > old_error[1]):
                    weight[1] = weight[1]/2
                if (abs(old_error[1] - error[1]) < 1e-6*error[1]):
                    if (not self.silent):
                        print("Reset weight for numerator.\n")
                    weight[1] = self.initial_weight
                self.G11n = (1-weight[1])*self.G11n_old + weight[1]*self.G11n
                self.G33n = (1-weight[1])*self.G33n_old + weight[1]*self.G33n

            if (not self.didconverge[0]):
                self.Sigma33d, self.Sigma11d = self.__get_Sigma(self.G33d,self.G11d)
                self.G33d_old = self.G33d
                self.G11d_old = self.G11d
                self.G33d, self.G11d = self.__get_G(self.Sigma33d, self.Sigma11d, self.G33_d_free_inverse, self.G11_d_free_inverse)
                error[0] = self.__get_error(self.G33d,self.G11d,self.G33d_old,self.G11d_old)
                if (error[0] > old_error[0]):
                    weight[0] = weight[0]/2
                if (abs(old_error[0] - error[0]) < 1e-6*error[0] ):
                    if (not self.silent):
                        print("Reset weight for denominator.\n")
                    weight[0] = self.initial_weight
                self.G11d = (1-weight[0])*self.G11d_old + weight[0]*self.G11d
                self.G33d = (1-weight[0])*self.G33d_old + weight[0]*self.G33d

            if (not self.silent):
                print(str(i)+ ". " +str(error))
            old_error[:] = error[:]










