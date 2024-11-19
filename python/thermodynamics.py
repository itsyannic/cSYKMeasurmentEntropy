import numpy as np
import cSYK
from scipy.optimize import fsolve
import json
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial
import os

maxQerr = 1e-5

def cSYKresults(beta,mu,q):
        
    results = cSYK.solve(beta,mu,q=q,reset=True)
    return [results['Q'], results['F']]

def Charge(xm,xb,q):

        result = cSYKresults(xb,xm,q)
        #print("Q=" +str(result[0]) + ", mu" + str(xm))
        return result[0]

def makeData(N_Q,N_beta,q):

    cSYK.init(30,0)

    Qs = np.linspace(0,0.30,N_Q,dtype=np.double,endpoint=False)
    betas = np.linspace(20,40,N_beta,dtype=np.double,endpoint=False)

    FreeArray = np.empty((N_beta,N_Q),dtype=np.double)
    QArray = np.empty((N_beta,N_Q),dtype=np.double)
    BetaArray =  np.empty((N_beta,N_Q),dtype=np.double)
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
            MuArray[j,i] = mu
            print(str(i*N_beta+j) + ": DQ=" + str(delta) + " and mu=" + str(mu) + " at beta=" + str(beta) + " and target Q=" + str(Qt))
            j+=1
        i+=1

    output = {"beta": BetaArray.tolist(), "Q": QArray.tolist(), "F": FreeArray.tolist(), "mu": MuArray.tolist()}
    json_obj = json.dumps(output)
    f = open("free_energy.json", "w")
    f.write(json_obj)
    f.close()

    sound = "/System/Library/Sounds/Submarine.aiff"
    os.system("afplay " + sound)

    return

def repairData(N_Q,N_beta,q):
    f = open("free_energy.json", "r")
    input = f.read()
    results = json.loads(input)
    betas = np.array(results["beta"])
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
                for p in range(1,4):
                    mu = fsolve(lambda x: Charge(x,betas[j,i],q)-Qtarget[i],mu,xtol=1e-7/(10**p))[0]
                    results = cSYKresults(betas[j,i],mu,q)
                    delta = abs(Qtarget[i]-results[0])
                    if (delta < maxQerr):
                        break
                
                Fs[j,i] = results[1]
                Qs[j,i] = results[0]
                mus[j,i] = mu
                count += 1

    output = {"beta": betas.tolist(), "Q": Qs.tolist(), "F": Fs.tolist(), "mu": mus.tolist()}
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
    Qs = np.array(results["Q"])
    Fs = np.array(results["F"])

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
    QPoly = []
    for Fvec in Fs:
        QPoly.append(polynomial.polyfit(Qs[i],Fvec,5))
        plt.scatter(Qs[i],Fvec,label="$\\beta=" + str(betas[i,0])+"$")
        pol = polynomial.Polynomial(QPoly[-1])
        plt.plot(Qs[i],pol(Qs[i]))
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$F$")
    plt.savefig('F_const_beta_lines.pdf',dpi=1000)
    plt.show()

    i=0
    BPoly = []
    for Fvec in Fs.swapaxes(0,1):
        BPoly.append(polynomial.polyfit(betas.swapaxes(0,1)[i],Fvec,5))
        plt.scatter(betas.swapaxes(0,1)[i],Fvec,label="$Q=" + str(Qs[0,i])+"$")
        pol = polynomial.Polynomial(BPoly[-1])
        plt.plot(betas.swapaxes(0,1)[i],pol(betas.swapaxes(0,1)[i]))
        i+=1
    plt.xlabel("$\\beta$")
    plt.ylabel("$F$")
    plt.savefig('F_const_Q_lines.pdf',dpi=1000)
    plt.show()

    i=0
    kappa_inv = []
    for pol in QPoly:
        kappa_inv.append(polynomial.Polynomial(pol).deriv(2))
        derivative = kappa_inv[-1]

        plt.plot(Qs[i],derivative(Qs[i]),label="$\\beta=" + str(betas[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\kappa^{-1}$")
    plt.savefig('kappa_inv.pdf',dpi=1000)
    plt.show()

    i=0
    gamma = []
    for pol in BPoly:
        gamma.append(-polynomial.Polynomial(pol).deriv(2))
        derivative = gamma[-1]
    
        plt.plot(betas.swapaxes(0,1)[i],derivative(betas.swapaxes(0,1)[i]),label="$Q=" + str(Qs[0,i])+"$")
        i+=1
    plt.xlabel("$\\beta$")
    plt.ylabel("$\\gamma$")
    plt.savefig('gamma.pdf',dpi=1000)
    plt.show()

    i=0
    mu_num = []
    for pol in QPoly:
        mu_num.append(polynomial.Polynomial(pol).deriv(1))
        derivative = mu_num[-1]

        plt.plot(Qs[i],derivative(Qs[i]),label="$\\beta=" + str(betas[i,0])+"$")
        i+=1
    plt.xlabel("$\\mathcal{Q}$")
    plt.ylabel("$\\mu$")
    plt.savefig('mu.pdf',dpi=1000)
    plt.show()

    return

if __name__ == "__main__":

    q = 4

    N_beta = 20
    N_Q = 30

    repairData(N_Q,N_beta,q)
