import numpy as np
from matplotlib import pyplot as plt
from SchwingerDyson import SchwingerDyson
import fields

beta = 1000
q = 4
J = 1
N = 100
L = 0.000001

sd = SchwingerDyson(beta,q,J,1,N,L,weight=0.5,max_iter=5000, silent=True)
sd.solve()
Gdij = fields.read_G_from_Ghat(sd.Ghatn, int(sd.discretization/2))
Gm = Gdij['G11']
Gmt = Gdij['G11'].transpose()
#print(sd.Ghatn)
#print(Gm)

tau1 = 0
tau2 = np.linspace(-2*beta,2*beta,4*N)
plt.plot(tau2,np.concatenate((-Gm[tau1], Gm[tau1])))
plt.show()