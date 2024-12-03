import numpy as np
import cSYK
from scipy.optimize import fsolve
import json
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial
import os

maxQerr = 1e-7

def kappa_inv_largeq(beta,Q,q):
    e = np.log((1-2*Q)/(1+2*Q))/(2*np.pi) + 2*np.pi*Q/(q**2)
    J = q/np.sqrt( 2*(2+2*np.cosh(2*np.pi*e))**(q/2-1) )
    return 4/(beta*(1-4*Q**2))+(16*J-4*np.pi**2/beta)/(q**2)


def cSYKresults(beta,mu,q):
        
    results = cSYK.solve(beta,mu,q=q,reset=True)
    return [results['Q'], results['F']]

def Charge(xm,xb,q):

        result = cSYKresults(xb,xm,q)
        #print("Q=" +str(result[0]) + ", mu" + str(xm))
        return result[0]

def makeData(N_Q,N_beta,q,invertbeta=True):

    cSYK.init(30,0)

    Qs = np.linspace(0,0.30,N_Q,dtype=np.double,endpoint=False)
    betas = np.linspace(20,40,N_beta,dtype=np.double,endpoint=False)
    Ts = np.reciprocal(betas)
    if invertbeta:
        Ts = np.linspace(0.02,0.04,N_beta,dtype=np.double,endpoint=False)
        betas = np.reciprocal(Ts)

    FreeArray = np.empty((N_beta,N_Q),dtype=np.double)
    QArray = np.empty((N_beta,N_Q),dtype=np.double)
    BetaArray =  np.empty((N_beta,N_Q),dtype=np.double)
    TArray = np.empty((N_beta,N_Q),dtype=np.double)
    MuArray = np.empty((N_beta,N_Q),dtype=np.double)

    i = 0
    muQ = 0
    for Qt in Qs:
        j=0
        mu = muQ
        for beta in betas:
            for p in range(3):
                mu = fsolve(lambda x: Charge(x,beta,q)-Qt,mu,xtol=1e-7/(10**p))[0]
                results = cSYKresults(beta,mu,q)
                delta = Qt-results[0]
                if (abs(delta) < maxQerr):
                    break
                print("redoing point, delta was: " + str(delta))
            if (j==0):
                 muQ=mu
            
            FreeArray[j,i] = results[1]
            QArray[j,i] = results[0]
            BetaArray[j,i] = beta
            TArray[j,i] = 1/beta
            MuArray[j,i] = mu
            print(str(i*N_beta+j) + ": DQ=" + str(delta) + " and mu=" + str(mu) + " at beta=" + str(beta) + " and target Q=" + str(Qt))
            j+=1
        i+=1

    output = {"beta": BetaArray.tolist(), "T": TArray.tolist(), "Q": QArray.tolist(), "F": FreeArray.tolist(), "mu": MuArray.tolist()}
    json_obj = json.dumps(output)
    f = open("free_energy.json", "w")
    f.write(json_obj)
    f.close()

    sound = "/System/Library/Sounds/Submarine.aiff"
    os.system("afplay " + sound)

    return

def repairData(N_Q,N_beta,q):

    cSYK.init(30,0)

    f = open("free_energy.json", "r")
    input = f.read()
    results = json.loads(input)
    betas = np.array(results["beta"])
    Ts = np.array(results["T"])
    Qs = np.array(results["Q"])
    Fs = np.array(results["F"])
    mus = np.array(results["mu"])

    Qtarget = np.linspace(0,0.30,N_Q,dtype=np.double,endpoint=False)

    mu = 0
    count = 0
    for i in range(N_Q):
        for j in range(N_beta):
            delta = abs(Qs[j,i] - Qtarget[i])
            if (delta > maxQerr):
                for p in range(1,6):
                    mu = fsolve(lambda x: Charge(x,betas[j,i],q)-Qtarget[i],mu,xtol=1e-7/(10**p))[0]
                    results = cSYKresults(betas[j,i],mu,q)
                    delta = abs(Qtarget[i]-results[0])
                    if (delta < maxQerr):
                        break
                
                Fs[j,i] = results[1]
                Qs[j,i] = results[0]
                mus[j,i] = mu
                count += 1

    output = {"beta": betas.tolist(), "T": Ts.tolist(), "Q": Qs.tolist(), "F": Fs.tolist(), "mu": mus.tolist()}
    json_obj = json.dumps(output)
    f = open("free_energy.json", "w")
    f.write(json_obj)
    f.close()
    print("corrected " + str(count) + " points")

    return

def processData(N_Q,N_beta,q):

    f = open("free_energy.json", "r")
    input = f.read()
    results = json.loads(input)
    betas = np.array(results["beta"])
    Ts = np.reciprocal(betas)
    Qs = np.array(results["Q"])
    Omegas = np.array(results["F"])
    mus = np.array(results["mu"])

    degBeta = 2
    degQ = 4

    kappa_inv_coef = np.empty((N_beta,degQ-1),dtype=np.double)
    mu_coef = np.empty((N_beta,degQ),dtype=np.double)
    gamma_coef = np.empty((N_Q,degBeta-1),dtype=np.double)

    Fs = Omegas + mus*Qs + mus/2

    Qtarget = np.linspace(0,0.30,N_Q,dtype=np.double,endpoint=False)
    i=0
    count = 0
    for Ql in Qs.reshape(-1):
         delta = Ql-Qtarget[i%N_Q]
         if (abs(delta) > maxQerr):
            print(delta)

            count+=1
         i+=1
    print(count)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 15

    i=0
    for OmegaVec in Omegas:
        plt.scatter(Qs[i],OmegaVec,label="$T=" + str(Ts[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\Omega$")
    plt.savefig('omega_const_beta_lines.pdf',dpi=1000)
    plt.show()

    i=0
    QPoly = []
    for Fvec in Fs:
        QPoly.append(polynomial.polyfit(Qs[i],Fvec,5))
        plt.scatter(Qs[i],Fvec,label="$T=" + str(Ts[i,0])+"$")
        pol = polynomial.Polynomial(QPoly[-1])
        plt.plot(Qs[i],pol(Qs[i]))
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$F$")
    plt.savefig('F_const_beta_lines.pdf',dpi=1000)
    plt.show()

    i=0
    TPoly = []
    for Fvec in Fs.swapaxes(0,1):
        TPoly.append(polynomial.polyfit(Ts.swapaxes(0,1)[i],Fvec,degBeta))
        plt.scatter(Ts.swapaxes(0,1)[i],Fvec,label="$Q=" + str(Qs[0,i])+"$")
        pol = polynomial.Polynomial(TPoly[-1])
        plt.plot(Ts.swapaxes(0,1)[i],pol(Ts.swapaxes(0,1)[i]))
        i+=1
    plt.xlabel("$T$")
    plt.ylabel("$F$")
    plt.savefig('F_const_Q_lines.pdf',dpi=1000)
    plt.show()

    i=0
    kappa_inv = []
    for pol in QPoly:
        kappa_inv.append(polynomial.Polynomial(pol).deriv(2))
        derivative = kappa_inv[-1]

        plt.plot(Qs[i],derivative(Qs[i]),label="$T=" + str(Ts[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\kappa^{-1}$")
    plt.savefig('kappa_inv_1.pdf',dpi=1000)
    plt.show()

    i=0
    gamma = []
    for pol in TPoly:
        gamma.append(-polynomial.Polynomial(pol).deriv(2))
        gamma_coef[i] = gamma[-1].coef
        derivative = gamma[-1]
    
        plt.plot(Ts.swapaxes(0,1)[i],derivative(Ts.swapaxes(0,1)[i]),label="$Q=" + str(Qs[0,i])+"$")
        i+=1
    plt.xlabel("$T$")
    plt.ylabel("$\\gamma$")
    plt.savefig('gamma.pdf',dpi=1000)
    plt.show()

    i=0
    mu_num = []
    for pol in QPoly:
        mu_num.append(polynomial.Polynomial(pol).deriv(1))
        derivative = mu_num[-1]

        plt.plot(Qs[i],derivative(Qs[i]),label="$T=" + str(Ts[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\mu$")
    plt.savefig('mu.pdf',dpi=1000)
    plt.show()

    i=0
    QPoly = []
    for muvec in mus:
        QPoly.append(polynomial.polyfit(Qs[i],muvec,degQ-1))
        plt.scatter(Qs[i],muvec,label="$T=" + str(Ts[i,0])+"$")
        mu_coef[i] = QPoly[-1]
        pol = polynomial.Polynomial(QPoly[-1])
        plt.plot(Qs[i],pol(Qs[i]))
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\mu$")
    plt.savefig('mu_const_beta_lines.pdf',dpi=1000)
    plt.show()

    i=0
    kappa_inv = []
    for pol in QPoly:
        kappa_inv.append(polynomial.Polynomial(pol).deriv(1))
        kappa_inv_coef[i] = kappa_inv[-1].coef
        derivative = kappa_inv[-1]

        plt.plot(Qs[i],derivative(Qs[i]),label="$T=" + str(Ts[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\kappa^{-1}$")
    plt.savefig('kappa_inv_2.pdf',dpi=1000)
    plt.show()
    
    i = int(N_beta/2)
    kappa_inv_theory = kappa_inv_largeq(betas[i,0],Qs[i],q)
    derivative = kappa_inv[int(N_beta/2)]
    plt.plot(Qs[i],derivative(Qs[i]),label="numerical")
    plt.plot(Qs[i],kappa_inv_theory,label="large $q$")
    plt.title("$T=" + str(Ts[i,0])+"$")
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\kappa^{-1}$")
    plt.legend()
    plt.savefig('kappa_inv_'+str(betas[i,0])+'=30.pdf',dpi=1000)
    plt.show()

    mu = np.empty((degQ,degBeta+1),dtype=np.double)
    for i in range(degQ):
        pol = polynomial.polyfit(Ts[:,0],mu_coef[:,i],degBeta)
        mu[i] = pol
        plt.scatter(Ts[:,0],mu_coef[:,i],label="c"+str(i))
        plt.plot(Ts[:,0],polynomial.Polynomial(pol)(Ts[:,0]))
    plt.title("$\\mu$ coefficients")
    plt.legend()
    plt.xlabel("$T$")
    plt.savefig('mu_coef.pdf',dpi=1000)
    plt.show()


    kappa_inv = np.empty((degQ-1,degBeta+1),dtype=np.double)
    for i in range(degQ-1):
        pol = polynomial.polyfit(Ts[:,0],kappa_inv_coef[:,i],degBeta)
        kappa_inv[i] = pol
        plt.scatter(Ts[:,0],kappa_inv_coef[:,i],label="c"+str(i))
        plt.plot(Ts[:,0],polynomial.Polynomial(pol)(Ts[:,0]))
    plt.title("$\\kappa^{-1}$ coefficients")
    plt.legend()
    plt.xlabel("$T$")
    plt.savefig('kappa_inv_coef.pdf',dpi=1000)
    plt.show()

    gamma = np.empty((degBeta-1,degQ+1),dtype=np.double)
    for i in range(degBeta-1):
        pol = polynomial.polyfit(Qs[0,:],gamma_coef[:,i],degQ)
        gamma[i] = pol
        plt.scatter(Qs[0,:],gamma_coef[:,i],label="c"+str(i))
        plt.plot(Qs[0,:],polynomial.Polynomial(pol)(Qs[0,:]))
    plt.title("$\\gamma$ coefficients")
    plt.legend()
    plt.xlabel("$\\mathcal{Q}$")
    plt.savefig('gamma_coef.pdf',dpi=1000)
    plt.show()


    output = {"gamma": gamma.tolist(), "kappa_inv": kappa_inv.tolist(), "mu": mu.tolist()}
    json_obj = json.dumps(output)
    f = open("gamma_kappa_coef.json", "w")
    f.write(json_obj)
    f.close()

    return

if __name__ == "__main__":

    q = 4

    N_beta = 20
    N_Q = 30

    processData(N_Q,N_beta,q)
