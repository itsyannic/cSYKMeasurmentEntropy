import numpy
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(1,1,1,0,10)

print(fields.convert_map_to_matrix(fields.G11,sd.Ghatn,10))

print(sd.Ghatn)