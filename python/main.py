import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics

ms = [0.0]
q = 8
beta = 50
J = 1

results = []

for m in ms:
    sd = SchwingerDyson(beta,q,J,m,200,0.00000001,weight=0.005)
    sd.solve()
    results.append(physics.results(sd))
    print(m)


param = {'q': q, 'beta': beta, 'J': J, 'data': results}
json_obj = json.dumps(param)
output = open("data.out", "w")

output.write(json_obj)