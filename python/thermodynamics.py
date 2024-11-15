import numpy as np
import cSYK
from scipy.optimize import fsolve

_dQ = np.double(1e-2)
_dB = np.double(1e-1)
cSYK.init(30,0)

def cSYKresults(beta,mu,q):
        
    results = cSYK.solve(beta,mu,q=q,reset=True)
    return [results['Q'], results['F']]

def Charge(xm,xb,q):

        result = cSYKresults(xb,xm,q)
        #print("Q=" +str(result[0]) + ", mu" + str(xm))
        return result[0]

def _gamma(mu,Q,beta,q,F):

    mulocal = fsolve(lambda x: Charge(x,beta+_dB,q)-Q,mu,xtol=1e-6)
    Finc = cSYKresults(beta+_dB,mulocal,q)[1]
    mulocal = fsolve(lambda x: Charge(x,beta-_dB,q)-Q,mu,xtol=1e-6)
    Fdec = cSYKresults(beta-_dB,mulocal,q)[1]

    return -(Finc-2*F+Fdec)/_dB**2

def _Kinv(mu,Q,beta,q,F):

    mulocal = fsolve(lambda x: Charge(x,beta,q)-Q-_dQ,mu,xtol=1e-6)
    Finc = cSYKresults(beta,mulocal,q)[1]
    mulocal = fsolve(lambda x: Charge(x,beta,q)-Q+_dQ,mu,xtol=1e-6)
    Fdec = cSYKresults(beta,mulocal,q)[1]

    return (Finc-2*F+Fdec)/_dQ**2

def _main():

    mu = 0.05
    beta = 30
    q = 4
    result = cSYKresults(beta,mu,q)
    FreeEnergy = result[1]
    Ql = result[0]
    print("F(Q=" + str(Ql) + ")=" + str(FreeEnergy))
    mulocal = fsolve(lambda x: Charge(x,beta,q)-Ql,mu,xtol=1e-7)
    Flocal = cSYKresults(beta,mulocal,q)[1]

    print(_Kinv(mu,Ql,beta,q,FreeEnergy))
    print(_gamma(mu,Ql,beta,q,FreeEnergy))

    return

if __name__ == "__main__":

    _main()
