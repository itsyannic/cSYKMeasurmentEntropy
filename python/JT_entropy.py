import numpy as np
import scipy as sc

def beta_times_curly_J(q,beta,e,J):

    return q*J/np.sqrt(2*(2+2*np.cosh(2*np.pi*e))**(q/2.0-1.0))*beta

def theta(Q,q):

    theta = sc.optimize.fsolve(lambda x: Q + x/np.pi+(0.5-1/q)*np.sin(2*x)/np.sin(2*np.pi/q), [-np.pi/q,np.pi/q])

    for t in theta:
        if (t>-np.pi/q) and (t<np.pi/q):
            return t
        return 0
    
def _curly_E(Q,m,q,beta):
    t = theta(Q,q)
    return np.log(np.sin(np.pi/q+t)/np.sin(np.pi/q-t))/(2*np.pi)

def integrand(x,epsilon):
    return 2*np.pi*x*np.sin(2*np.pi*x)/(np.cosh(2*np.pi*epsilon)-np.cos(2*np.pi*x))

def _S_0(Q,q,e):

    return 0.47138 + sc.integrate.quad(lambda x: 2*np.pi*_curly_E(x,1,q,1), 0, Q)[0]


def _S_Gravity(Q,q,beta_x_curlyJ,e):

    return np.pi**2*(q-2)*np.tanh(np.pi*e)/(2*beta_x_curlyJ*q**2*(1-4*Q**2))

def _S_Gauge(Q,q,beta_x_curlyJ,e):

    return (q*np.pi*e/2)**2/beta_x_curlyJ*(1+e**2/beta_x_curlyJ*(4+np.pi*e*(q-2)*np.tanh(np.pi*e))/(4+16/q**2*beta_x_curlyJ*(1-4*Q**2+Q*(q-2)*np.tanh(np.pi*e))))

def S_IR(Qin,m,q,beta,J):
    Q=Qin
    e = _curly_E(Q,m,q,beta)
    betaJ = beta_times_curly_J(q,beta,e,J)

    return _S_Gauge(Q,q,betaJ,e) + _S_Gravity(Q,q,betaJ,e) + _S_0(Q,q,e) -m*2*np.log(2.0)/12.0

def S_UV(m,q):

    k = 1-m

    return k*(np.log(2)-1/q**2*(np.arcsin(k**(3/2)))**2)

def S_gen(Q,m,q,beta,J):

    return np.max([np.min([S_UV(m,q), S_IR(Q,m,q,beta,J)]),0])
    