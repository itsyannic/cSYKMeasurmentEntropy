import numpy as np
from SchwingerDyson import SchwingerDyson

sd = SchwingerDyson(100,8,1,0,500,0.000000001)
sd.solve()

print(np.trace(sd.Ghatn))