import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics

ms = [0,0.5,0.9]
q = 8
beta = 100
J = 1

results = []

for m in ms:
    print(m)
    sd = SchwingerDyson(beta,q,J,m,100,0.00000001)
    sd.solve()
    results.append(physics.results(sd))


param = {'q': q, 'beta': beta, 'J': J, 'data': results}
json_obj = json.dumps(param)
output = open("data.out", "w")

output.write(json_obj)