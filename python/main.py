import sys
import os
import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy
from matplotlib import pyplot as plt

#set plot parameters
generate_data = False
ms = np.linspace(0,1.0,50,endpoint=False, dtype=np.double)
q = 4
beta = 30
N = 400
L = np.double(0.00000000001)
J = 1

#process system input and re-set beta if required
if (len(sys.argv) > 1):
    beta = np.double(sys.argv[1])

print("beta = " +str(beta))

#prepare output
filebase = 'Data/beta=' + str(beta) + 'q=' + str(q) + 'N=' +str(N)

if (generate_data):
    
    #generate numerical data
    sd = SchwingerDyson(beta,q,J,0,N,L,weight=0.5,max_iter=5000, silent=True)
    results = []

    for m in ms:
        sd.reset()
        sd.m = m
        sd.solve()

        results.append(physics.results(sd))
        print(str(m) + ": S_JT=" +str(JT_entropy.S_gen(results[-1]['charge'],m,q,beta,J)) + ", S_cSYK=" + str(results[-1]['renyi2']) + ",  tr(G33)=" + str(results[-1]['trG33']))

    #save data to file
    param = {'q': q, 'beta': beta, 'J': J, 'N': N, 'L': L, 'data': results}
    json_obj = json.dumps(param)
    output = open(filebase + '_data.out', "w")
    output.write(json_obj)
    output.close()
else:
    file = open(filebase + '_data.out', "r")
    input = file.read()
    results = json.loads(input)['data']
    file.close()


#plot data
m = [point['m'] for point in results]
I = [point['renyi2'] for point in results]
x = np.linspace(0,1,50, endpoint=False)
JT = [JT_entropy.S_gen(point['charge'],point['m'],q,beta,J) for point in results]

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15

plt.scatter(m,I,label="complex SYK")
plt.plot(m,JT,label="charged JT")
plt.xlabel('$m$')
plt.ylabel('$S$')
plt.title('$\\beta = ' + str(beta) + '$, $q=' + str(q) +'$')
plt.legend(loc="upper right")
plt.savefig(filebase + '.pdf',dpi=1000)
plt.show()

sound = "/System/Library/Sounds/Submarine.aiff"
os.system("afplay " + sound)