import numpy as np
import scipy as sc

def beta_times_curly_J(q,beta,e,J):

    return q*J/np.sqrt(2*(2+2*np.cosh(2*np.pi*e))**(q/2.0-1.0))*beta

def theta(Q,q):

    theta = sc.optimize.fsolve(lambda x: Q + x/np.pi+(0.5-1/q)*np.sin(2*x)/np.sin(2*np.pi/q), [-np.pi/q,np.pi/q], xtol=1e-16)

    for t in theta:
        if (t>-np.pi/q) and (t<np.pi/q):
            return t
        return 0
    
def _curly_E(Q,q):
    t = theta(Q,q)
    return np.log(np.sin(np.pi/q+t)/np.sin(np.pi/q-t))/(2*np.pi)

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

    return (q*np.pi*e/2)**2/beta_x_curlyJ*(1+e**2/beta_x_curlyJ*(4+np.pi*e*(q-2)*np.tanh(np.pi*e))/(4+16/q**2*beta_x_curlyJ*(1-4*Q**2+Q*(q-2)*np.tanh(np.pi*e))))

def S_IR(Qin,m,q,beta,J):
    Q=Qin
    e = _curly_E(Q,q)
    betaJ = beta_times_curly_J(q,beta,e,J)

    return _S_Gauge(Q,q,betaJ,e) + _S_Gravity(Q,q,betaJ,e) + _S_0(Q,q,e) -m*2*np.log(2.0)/12.0

def S_UV(m,q):

    k = 1-m

    return k*(np.log(2)-1/q**2*(np.arcsin(k**(3/2)))**2)

def S_gen(Q,m,q,beta,J):

    return np.max([np.min([S_UV(m,q), S_IR(Q,m,q,beta,J)]),0])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Qx = np.linspace(0,0.49,100)
    S_JT = np.array([_S_JT(Qx[i],30,4,1) for i in range(len(Qx))])
    S_large_q = np.array([_S_Gauge(Qx[i],4,30,_curly_E(Qx[i],4)) + _S_Gravity(Qx[i],4,30,_curly_E(Qx[i],4)) for i in range(len(Qx))])

    plt.plot(Qx,S_JT)
    plt.plot(Qx,S_large_q)
    plt.show()
    