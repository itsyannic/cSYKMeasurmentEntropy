import numpy as np
from matplotlib import pyplot as plt
from SchwingerDyson import SchwingerDyson
import fields

beta = 1000
q = 4
J = 0
N = 100
L = 0.000001

sd = SchwingerDyson(beta,q,J,1,N,L,weight=0.5,max_iter=5000, silent=True)
sd.solve()
print(np.log(np.linalg.det(sd.Ghat_n_free_inverse))-np.log(np.linalg.det(sd.Ghat_d_free_inverse)))
print(np.log(np.linalg.det(sd.G33_n_free_inverse))-np.log(np.linalg.det(sd.G33_d_free_inverse)))
Gdij = fields.read_G_from_Ghat(sd.Ghatn, int(sd.discretization/2))
Gm = Gdij['G11']
#Gm = sd.G33n
Gmt = Gdij['G11'].transpose()

tau1 = 0
tau2 = np.linspace(-2*beta,2*beta,4*N)
plt.plot(tau2,np.concatenate((-Gm[tau1], Gm[tau1])))
plt.show()