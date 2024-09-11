import numpy as np
from matplotlib import pyplot as plt
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy

files = ['beta=2q=4N=80_data', 'beta=5q=4N=10_data', 
         'beta=10q=4N=80_data', 'beta=20q=4N=20_data',
         'beta=40q=4N=20_data', 'beta=50q=4N=20_data']

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
    x = np.linspace(0,1,50, endpoint=False)

    plt.scatter(Q,I, label='beta='+str(data['beta']))

plt.xlabel('m')
plt.ylabel('S')
plt.legend()
plt.title('Renyi-2 Entropy for q=4')
plt.savefig('entropyvscharge.jpg')
plt.show()