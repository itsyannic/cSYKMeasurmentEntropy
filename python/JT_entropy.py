import numpy as np
import scipy as sc
import json

def kappa_inv_largeq(beta,Q,q):
    e = np.log((1-2*Q)/(1+2*Q))/(2*np.pi) + 2*np.pi*Q/(q**2)
    J = q/np.sqrt( 2*(2+2*np.cosh(2*np.pi*e))**(q/2-1) )
    return 4/(beta*(1-4*Q**2))+(16*J-4*np.pi**2/beta)/(q**2)

def mu_largeq(beta,Q,q):

    e = np.log((1-2*Q)/(1+2*Q))/(2*np.pi) + 2*np.pi*Q/(q**2)
    J = q/np.sqrt( 2*(2+2*np.cosh(2*np.pi*e))**(q/2-1) )

    return 16*J*Q/q**2 - 2*np.pi*e/beta

def gamma_largeq(beta,Q,q):

    e = np.log((1-2*Q)/(1+2*Q))/(2*np.pi) + 2*np.pi*Q/(q**2)
    J = q/np.sqrt( 2*(2+2*np.cosh(2*np.pi*e))**(q/2-1) )

    return 2*np.pi**2*(1-4*Q**2)/(J*q**2)

class cSYK_termodynamics:
    init = 0

    _kappa_inv_coef = []
    _gamma_coef = []
    _mu_coef = []

    _deg_kappa = 0
    _deg_gamma = 0
    _deg_mu = 0

    @classmethod
    def _initialize(cls):

        f = open("gamma_kappa_coef.json", "r")
        input = f.read()
        coefs = json.loads(input)
        gamma_arr = np.array(coefs["gamma"])
        kappa_inv_arr = np.array(coefs["kappa_inv"])
        mu_arr = np.array(coefs["mu"])

        for vector in mu_arr:
            pol = np.polynomial.polynomial.Polynomial(vector)
            cls._mu_coef.append(pol)

        cls._deg_mu = len(cls._mu_coef)

        for vector in kappa_inv_arr:
            pol = np.polynomial.polynomial.Polynomial(vector)
            cls._kappa_inv_coef.append(pol)

        cls._deg_kappa = len(cls._kappa_inv_coef)

        for vector in gamma_arr:
            pol = np.polynomial.polynomial.Polynomial(vector)
            cls._gamma_coef.append(pol)

        cls._deg_gamma = len(cls._gamma_coef)

        cls.init = 1
        return
    
    @classmethod
    def mu(cls,Q,beta):

        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._mu_coef:
            coef.append(polynomial(1/beta))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(Q)
    
    @classmethod
    def dmu_dT(cls,Q,beta):

        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._mu_coef:
            der = polynomial.deriv(1)
            coef.append(der(1/beta))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(Q)
    
    @classmethod
    def kappa_inv(cls,Q,beta):

        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._kappa_inv_coef:
            coef.append(polynomial(1/beta))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(Q)
    
    @classmethod
    def dkappa_inv_dT(cls,Q,beta):

        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._kappa_inv_coef:
            der = polynomial.deriv(1)
            coef.append(der(1/beta))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(Q)

    @classmethod
    def gamma(cls,Q,beta):
        
        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._gamma_coef:
            coef.append(polynomial(Q))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(1/beta)
    

def _beta_times_curly_J(q,beta,e,J):

    return q*J/np.sqrt(2*(2+2*np.cosh(2*np.pi*e))**(q/2.0-1.0))*beta
    
def _curly_E(Q,q):
    
    return -(8*np.pi*Q)/(3*q**3) + (2*np.pi*Q)/q**2 + ((16*Q*np.pi**3)/5 - (448*np.pi**3*Q**3)/15)/q**5 + (-((2*np.pi**3*Q)/3) + 8*np.pi**3*Q**3)/q**4 + np.log(-1 + 2/(1 + 2*Q))/(2*np.pi)

def _integrand(x,epsilon):
    return 2*np.pi*x*np.sin(2*np.pi*x)/(np.cosh(2*np.pi*epsilon)-np.cos(2*np.pi*x))

def _S_0(Q,q,e):

    return 0.435 + sc.integrate.quad(lambda x: 2*np.pi*_curly_E(x,q), 0, Q)[0]
    #return (2*np.pi**2*(1 - 4*Q**2))/(3*q**3) + (np.pi**2*(-1 + 4*Q**2))/(2*q**2) + (2*np.pi**4*(1 + 24*Q**2 - 112*Q**4))/(15*q**5) + (np.pi**4*(-1 - 8*Q**2 + 48*Q**4))/(12*q**4) + (1/2)*np.log(4/(1 - 4*Q**2)) + Q*np.log(-1 + 2/(1 + 2*Q))

def _S_JT(Q,beta,q,J):

    e = _curly_E(Q,q)

    gamma = cSYK_termodynamics.gamma(Q,beta)
    kappa = 1/cSYK_termodynamics.kappa_inv(Q,beta)
    dkappa = -cSYK_termodynamics.dkappa_inv_dT(Q,beta)*kappa**2
    dmu = cSYK_termodynamics.dmu_dT(Q,beta)

    I_JT = -gamma/(2*beta) - 2*np.pi**2*e**2*kappa/(beta)

    return -I_JT+(1/beta)*(gamma/2+2*np.pi**2*e**2*(kappa+dkappa/beta)-dmu*Q)

def S_IR(Qin,m,q,beta,J):
    Q=Qin

    return _S_JT(Q,beta,q,J) + _S_0(Q,q,0) -m*2*np.log(2.0)/12.0

def S_UV(Q,q):

    k = 1-2*Q

    return k*(np.log(2)-(1/q**2)*( np.arcsin(k**(3/2)) )**2 )

def S_gen(Q,m,q,beta,J):

    if (Q>0.35):
        return S_UV(Q,q)

    return np.min([S_UV(Q,q), S_IR(Q,m,q,beta,J)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 15  

    Qx = np.linspace(0,0.35,100)
    plt.plot(Qx, [cSYK_termodynamics.kappa_inv(Q,30) for Q in Qx],label="numerical")
    kappa_inv_theory = kappa_inv_largeq(30,Qx,4)
    plt.plot(Qx,kappa_inv_theory,label="large $q$")
    plt.title("$\\beta=30$")
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\kappa^{-1}$")
    plt.legend()
    plt.savefig('kappa_inv_beta=30.pdf',dpi=1000)
    plt.show()

    plt.plot(Qx, [cSYK_termodynamics.mu(Q,30) for Q in Qx],label="numerical")
    mu_theory = mu_largeq(30,Qx,4)
    plt.plot(Qx,mu_theory,label="large $q$")
    plt.ylabel("$\\mu$")
    plt.title("$\\beta=30$")
    plt.xlabel("$\\mathcal{Q}$")
    plt.legend()
    plt.savefig('mu_beta=30.pdf',dpi=1000)
    plt.show()

    plt.plot(Qx, [cSYK_termodynamics.gamma(Q,30) for Q in Qx],label="numerical")
    gamma_theory = gamma_largeq(30,Qx,4)
    plt.plot(Qx,gamma_theory,label="large $q$")
    plt.ylabel("$\\gamma$")
    plt.title("$\\beta=30$")
    plt.xlabel("$\\mathcal{Q}$")
    plt.legend()
    plt.savefig('gamma_beta=30.pdf',dpi=1000)
    plt.show()

    Qy = np.linspace(0,0.5,100)

    S_0 = np.array([_S_0(Q,4,1) for Q in Qy])
    S_JT = np.array([_S_JT(Q,30,4,1) for Q in Qx])
    S_IR = np.array([_S_JT(Q,30,4,1)+ _S_0(Q,4,1) for Q in Qx])
    S_UV = [S_UV(Q,4)for Q in Qy]

    plt.plot(Qx,S_JT, label="$S_\mathrm{IR}-S_0$",color="slategrey")
    plt.plot(Qx,S_IR, label="$S_\mathrm{IR}$",color="forestgreen")
    plt.plot(Qy,S_0, label="$S_0$",color="cornflowerblue")
    plt.plot(Qy,S_UV,label="$S_\mathrm{UV}$",color="lightseagreen")
    plt.ylabel("$S$")
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylim(0,0.6)
    plt.legend()
    plt.show()
    