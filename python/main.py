import numpy as np
from matplotlib import pyplot as plt
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy

#set plot parameters
ms = np.linspace(0,1,10,endpoint=False, dtype=np.double)
q = 4
beta = 50
J = 1

if (True):
    #generate numerical data
    results = []

    for m in ms:
        sd = SchwingerDyson(beta,q,J,m,80,0.001,weight=0.5,max_iter=10000)
        sd.solve()
        results.append(physics.results(sd))
        print(m)

    #save data to file
    param = {'q': q, 'beta': beta, 'J': J, 'data': results}
    json_obj = json.dumps(param)
    output = open("data.out", "w")
    output.write(json_obj)
    output.close()
else:
    file = open("data.out", "r")
    input = file.read()
    results = json.loads(input)['data']
    file.close()

#plot data

m = [point['m'] for point in results]
I = [point['renyi2'] for point in results]
#JT = [JT_entropy.S_gen(point['charge'],point['m'],q,beta,J) for point in results]


plt.scatter(m,I)
#plt.plot(m,JT)
plt.xlabel('m')
plt.ylabel('Renyi-2 Entropy I2')
plt.title('beta = ' + str(beta) + ', q=' + str(q))
plt.show()
