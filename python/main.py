import numpy as np
from matplotlib import pyplot as plt
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy

#set plot parameters
ms = np.linspace(0,0.5,20,endpoint=False, dtype=np.double)
#ms = [0.325]
q = 4
beta = 100
N = 400
steps = 0
target_beta = 80
J = 1


if (True):
    
    #generate numerical data
    sd = SchwingerDyson(beta,q,J,0,N,0.00000001,weight=0.5,max_iter=10000)
    results = []

    for m in ms:
        sd.m = m
        sd.reset()
        sd.solve()
        for i in range(steps):
            sd._beta = beta + (target_beta-beta)*(i/steps)
            sd.solve()
        results.append(physics.results(sd))
        print(m)
        print(JT_entropy.S_IR(0,m,q,beta,J))
        print(results[-1]['renyi2'])

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
JT = [JT_entropy.S_gen(point['charge'],point['m'],q,beta,J) for point in results]


plt.scatter(m,I)
plt.plot(m,JT)
plt.xlabel('Charge Q')
plt.ylabel('Renyi-2 Entropy I2')
plt.title('beta = ' + str(target_beta) + ', q=' + str(q))
plt.show()