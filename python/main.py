import numpy as np
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(100,8,1,0,100,0.0001)
sd.solve()

print(np.trace(sd.Ghatn))