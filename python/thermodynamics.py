import numpy as np
import cSYK
from scipy.optimize import fsolve
import json
from matplotlib import pyplot as plt

def cSYKresults(beta,mu,q):
        
    results = cSYK.solve(beta,mu,q=q,reset=True)
    return [results['Q'], results['F']]

def Charge(xm,xb,q):

        result = cSYKresults(xb,xm,q)
        #print("Q=" +str(result[0]) + ", mu" + str(xm))
        return result[0]

def makeData():

    q = 4

    cSYK.init(30,0)

    N_beta = 20
    N_Q = 30

    Qs = np.linspace(0,0.30,N_Q,dtype=np.double)
    betas = np.linspace(20,40,N_beta,dtype=np.double)

    FreeArray = np.empty((N_beta,N_Q),dtype=np.double)
    QArray = np.empty((N_beta,N_Q),dtype=np.double)
    BetaArray =  np.empty((N_beta,N_Q),dtype=np.double)

    i = 0
    muQ = 0
    for Ql in Qs:
        j=0
        mu = muQ
        for beta in betas:
            mu = fsolve(lambda x: Charge(x,beta,q)-Ql,mu,xtol=1e-7)[0]
            if (j==0):
                 muQ=mu
            results = cSYKresults(beta,mu,q)
            FreeArray[j,i] = results[1]
            QArray[j,i] = results[0]
            BetaArray[j,i] = beta
            print(Ql-results[0])
            j+=1
        i+=1

    output = {"beta": BetaArray.tolist(), "Q": QArray.tolist(), "F": FreeArray.tolist()}
    json_obj = json.dumps(output)
    f = open("free_energy.json", "w")
    f.write(json_obj)
    f.close()

    return

def processData():

    f = open("free_energy.json", "r")
    input = f.read()
    results = json.loads(input)
    betas = np.array(results["beta"])
    Qs = np.array(results["Q"])
    Fs = np.array(results["F"])

    N_Q = 30
    Qtarget = np.linspace(0,0.30,N_Q,dtype=np.double)
    i=0
    count = 0
    for Ql in Qs.reshape(-1):
         delta = Ql-Qtarget[i%N_Q]
         if delta > 1e-3:
            print(delta)

            count+=1
         i+=1
    print(count)

    i=0
    for Fvec in Fs.swapaxes(0,1):
        plt.scatter(betas.swapaxes(0,1)[i],Fvec)
        i+=1
    plt.show()
         
    return

if __name__ == "__main__":

    makeData()
