import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics

num_ms = 2
q = 8
beta = 100
J = 1

results = []

for i in range(num_ms):
    m = 1.0*(i/num_ms)

    sd = SchwingerDyson(beta,q,J,m,500,0.0000000001)
    sd.solve()
    results.append(physics.results(sd))


param = {'q': q, 'beta': beta, 'J': J, 'data': results}
json_obj = json.dumps(param)
output = open("data.out", "w")

output.write(json_obj)