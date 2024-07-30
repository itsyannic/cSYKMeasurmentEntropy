import numpy as np
from matplotlib import pyplot as plt
from SchwingerDyson import SchwingerDyson
import fields

beta = 20
q = 4
J = 1
N = 100
L = 0.000001

sd = SchwingerDyson(beta,q,J,1,N,L,weight=0.5,max_iter=5000, silent=True)
sd.solve()
Gdij = fields.read_G_from_Ghat(sd.Ghatn, int(sd.discretization/2))
Gm = Gdij['G11'] + Gdij['G22'] 
Gmt = Gm.transpose()

tau = np.linspace(-2*beta,2*beta,4*N)
plt.plot(tau,np.concatenate((-Gm[0],Gm[0])))
plt.show()