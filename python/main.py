import sys
import os
import numpy as np
import json
from SchwingerDyson import SchwingerDyson
import physics
import JT_entropy
from matplotlib import pyplot as plt

#set plot parameters
generate_data = True
ms = np.linspace(0,1.0,50,endpoint=False, dtype=np.double)
q = 4
beta = 20
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
        print(results[-1])

    #save data to file
    param = {'q': q, 'beta': beta, 'J': J, 'N': N, 'L': L, 'data': results}
    json_obj = json.dumps(param)
    if (not os.path.exists("Data/")):
        os.makedirs("Data")
else:
    file = open(filebase + '_data.out', "r")
    input = file.read()
    results = json.loads(input)['data']
    file.close()

sound = "/System/Library/Sounds/Submarine.aiff"
os.system("afplay " + sound)