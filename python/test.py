import numpy as np
from matplotlib import pyplot as plt
from SchwingerDyson import SchwingerDyson
import fields

beta = 20
q = 4
J = 0
N = 80
L = 0.000001

sd = SchwingerDyson(beta,q,J,1,N,L,weight=0.5,max_iter=5000, silent=True)
Gdij = fields.read_G_from_Ghat(sd.Ghatn, int(sd.discretization/2))
Gm = Gdij['G11']
Gmt = Gdij['G11'].transpose()

print(len(Gm[0]))
tau = np.linspace(-2*beta,2*beta,4*N)
plt.plot(tau,np.concatenate((Gmt[0], Gm[0])))
plt.show()