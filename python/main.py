import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics

ms = np.linspace(0,1,10,endpoint=False)
q = 8
beta = 50
J = 1

results = []

for m in ms:
    sd = SchwingerDyson(beta,q,J,m,200,0.0001,weight=0.0005,max_iter=10000)
    sd.solve()
    results.append(physics.results(sd))
    print(m)


param = {'q': q, 'beta': beta, 'J': J, 'data': results}
json_obj = json.dumps(param)
output = open("data.out", "w")

output.write(json_obj)