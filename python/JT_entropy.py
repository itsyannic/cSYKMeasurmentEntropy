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
    def gamma(cls,Q,beta):
        
        if (not cls.init):
            cls._initialize()
            cls.init = 1

        coef = []
        for polynomial in cls._gamma_coef:
            coef.append(polynomial(Q))

        polynomial = np.polynomial.polynomial.Polynomial(coef)

        return polynomial(1/beta)
    

def beta_times_curly_J(q,beta,e,J):

    return q*J/np.sqrt(2*(2+2*np.cosh(2*np.pi*e))**(q/2.0-1.0))*beta
    
def _curly_E(Q,q):
    
    return -(8*np.pi*Q)/(3*q^3) + (2*np.pi*Q)/q**2 + ((16*Q*np.pi**3)/5 - (448*np.pi**3*Q**3)/15)/q**5 + (-((2*np.pi**3*Q)/3) + 8*np.pi**3*Q**3)/q**4 + np.log(-1 + 2/(1 + 2*Q))/(2*np.pi)

def integrand(x,epsilon):
    return 2*np.pi*x*np.sin(2*np.pi*x)/(np.cosh(2*np.pi*epsilon)-np.cos(2*np.pi*x))

def _S_0(Q,q,e):

    return 0.47138 + sc.integrate.quad(lambda x: 2*np.pi*_curly_E(x,q), 0, Q)[0]

def _I_JT(Q,beta,q,J):
    e = _curly_E(Q,q)
    betaJ = beta_times_curly_J(q,beta,e,J)
    return -(q*np.pi*e)**2/(8*betaJ) - np.pi**2/(betaJ*q**2)

def _S_JT(Q,beta,q,J):

    dQ = np.double(1e-12)
    dbeta = np.double(1e-12)

    I_JT = _I_JT(Q,beta,q,J)
    dI_JT = [_I_JT(Q+dQ,beta,q,J),_I_JT(Q,beta+dbeta,q,J),_I_JT(Q+dQ,beta+dbeta,q,J),
             _I_JT(Q-dQ,beta,q,J),_I_JT(Q,beta-dbeta,q,J),_I_JT(Q-dQ,beta-dbeta,q,J),
             _I_JT(Q+dQ,beta-dbeta,q,J), _I_JT(Q-dQ,beta+dbeta,q,J)]
    F = -I_JT/beta
    dF = [dI_JT[0]/beta,0,dI_JT[2]/(beta+dbeta),
          dI_JT[3]/beta,0,dI_JT[5]/(beta-dbeta),
          dI_JT[6]/(beta-dbeta),dI_JT[7]/(beta+dbeta)]
    
    dIdQ = (dI_JT[0]-dI_JT[3])/(2*dQ)
    dIdB = (dI_JT[1]-dI_JT[4])/(2*dbeta)
    dFdQQ = (-dF[0]-dF[3]+2*F)/(dQ**2)
    dFdQdB = (dF[3]+dF[5]-dF[6]-dF[7])/(4*dQ*dbeta)

    return -I_JT+beta*(dIdB-dIdQ*dFdQdB/dFdQQ)

def _S_Gravity(Q,q,beta_x_curlyJ,e):

    return np.pi**2*(q-2)*np.tanh(np.pi*e)/(2*beta_x_curlyJ*q**2*(1-4*Q**2))

def _S_Gauge(Q,q,beta_x_curlyJ,e):

    return (q*np.pi*e/2)**2/beta_x_curlyJ*(1+e**2/beta_x_curlyJ*(4+np.pi*e*(q-2)*np.tanh(np.pi*e))/( 4+16/q**2*beta_x_curlyJ*(1-4*Q**2+Q*(q-2)*np.tanh(np.pi*e)) ))

def S_IR(Qin,m,q,beta,J):
    Q=Qin
    e = _curly_E(Q,q)
    betaJ = beta_times_curly_J(q,beta,e,J)

    return _S_JT(Q,beta,q,J) + _S_0(Q,q,e) -m*2*np.log(2.0)/12.0

def S_UV(Q,q):

    k = 1-2*Q

    return k*(np.log(2)-(1/q**2)*( np.arcsin(k**(3/2)) )**2 )

def S_gen(Q,m,q,beta,J):

    return np.min([S_UV(Q,q), S_IR(Q,m,q,beta,J)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

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
    plt.legend()
    plt.savefig('mu_beta=30.pdf',dpi=1000)
    plt.show()

    Qx = np.linspace(0,0.49,100)
    S_0 = np.array([_S_0(Qx[i],4,1) for i in range(len(Qx))])
    S_JT = np.array([_S_JT(Qx[i],30,4,1) for i in range(len(Qx))])
    S_large_q = np.array([_S_Gauge(Qx[i],4,30,_curly_E(Qx[i],4)) + _S_Gravity(Qx[i],4,30,_curly_E(Qx[i],4)) for i in range(len(Qx))])
    S_UV = [S_UV(Qx[i],4)for i in range(len(Qx))]

    plt.plot(Qx,S_JT, label="JT from I*",color="slategrey")
    plt.plot(Qx,S_0, label="S0",color="cornflowerblue")
    plt.plot(Qx,S_large_q,label="large q JT",color="navy")
    plt.plot(Qx,S_UV,label="UV",color="lightseagreen")
    plt.plot(Qx,np.zeros(len(Qx)),label="0",color="red")
    plt.legend()
    plt.show()
    