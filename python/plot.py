import numpy as np
from matplotlib import pyplot as plt
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy

plt.rcParams['text.usetex'] = True

files = ['beta=2.0q=4N=400_data', 'beta=5.0q=4N=400_data', 
         'beta=10.0q=4N=400_data', 'beta=20.0q=4N=400_data', 
         'beta=30.0q=4N=400_data', 'beta=40.0q=4N=400_data', 
         'beta=50.0q=4N=400_data']

x = np.linspace(0,1,50, endpoint=False)

for filename in files:
    print(filename)
    file = open('Data/' + filename + ".out", "r")
    input = file.read()
    data = json.loads(input)
    results = data['data']
    file.close()


    #plot data
    m = [point['m'] for point in results]
    Q = [point['charge'] for point in results]
    I = [point['renyi2'] for point in results]

    plt.plot(m,I, label='$\\beta=$'+str(data['beta']))

plt.xlabel('$m$')
plt.ylabel('$S$')
plt.ylim(0,0.6)
plt.xlim(0,1.0)
plt.legend()
plt.title('Reny-2 Entropy for $q=4$')
plt.savefig('entropyvsm.pdf', dpi=1000)
plt.show()

for filename in files:

    file = open('Data/' + filename + ".out", "r")
    input = file.read()
    data = json.loads(input)
    results = data['data']
    file.close()


    #plot data
    m = [point['m'] for point in results]
    Q = [point['charge'] for point in results]
    I = [point['renyi2'] for point in results]

    plt.scatter(Q,I, label='$\\beta=$'+str(data['beta']))

plt.xlabel('$\mathcal{Q}$')
plt.ylabel('$S$')
plt.xlim(0,0.5)
plt.ylim(0,0.6)
plt.legend()
plt.title('Reny-2 Entropy for $q=4$')
plt.savefig('entropyvsQ.pdf', dpi=1000)
plt.show()

for filename in files:

    file = open('Data/' + filename + ".out", "r")
    input = file.read()
    data = json.loads(input)
    results = data['data']
    file.close()


    #plot data
    m = [point['m'] for point in results]
    Q = [point['charge'] for point in results]
    I = [point['renyi2'] for point in results]

    plt.plot(m,Q, label='$\\beta=$'+str(data['beta']))

plt.xlabel('$m$')
plt.ylabel('$\mathcal{Q}$')
plt.ylim(0,0.6)
plt.xlim(0,1.0)
plt.legend()
plt.title('Charge $\mathcal{Q}$ for $q=4$')
plt.savefig('mvsQ.pdf', dpi=1000)
plt.show()