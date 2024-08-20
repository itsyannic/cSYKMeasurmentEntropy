import numpy as np
from matplotlib import pyplot as plt
from SchwingerDyson import SchwingerDyson
import fields
import physics

beta = 100
q = 4
J = 0
N = 200
L = 0.000001

sd = SchwingerDyson(beta,q,J,1,N,L,weight=0.5,max_iter=5000, silent=True)
sd.solve()
print(np.log(np.linalg.det(sd.G11_n_free_inverse))-np.log(np.linalg.det(sd.G11_d_free_inverse)))
print(np.log(np.linalg.det(sd.G33_n_free_inverse))-np.log(np.linalg.det(sd.G33_d_free_inverse)))
results = physics.results(sd)
print(results['Iden'])
print(results['Inum'])
print("S="+str(results['renyi2']))

Gm = sd.G11n
#Gm = sd.G33n
Gmt = Gm.transpose()

tau1 = 0
tau2 = np.linspace(-2*beta,2*beta,4*N)
plt.plot(tau2,np.concatenate((-Gm[tau1], Gm[tau1])))
plt.show()