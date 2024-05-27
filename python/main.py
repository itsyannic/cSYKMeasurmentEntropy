import numpy as np
from matplotlib import pyplot as plt
import json
from SchwingerDyson import SchwingerDyson
import physics

#set plot parameters
ms = np.linspace(0,1,10,endpoint=False)
q = 8
beta = 50
J = 1

#generate numerical data
results = []

for m in ms:
    sd = SchwingerDyson(beta,q,J,m,200,0.0001,weight=0.0005,max_iter=10000)
    sd.solve()
    results.append(physics.results(sd))
    print(m)

#save data to file
param = {'q': q, 'beta': beta, 'J': J, 'data': results}
json_obj = json.dumps(param)
output = open("data.out", "w")

#plot data
Q = [point['charge'] for point in results]
I = [point['renyi2'] for point in results]

plt.scatter(Q,I)
plt.xlabel('Charge Q')
plt.ylabel('Renyi-2 Entropy I2')
plt.title('beta = 50, q=8')
plt.show()

output.write(json_obj)