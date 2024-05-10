import numpy
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(1,1,1,0,4)

print(fields.convert_map_to_matrix(fields.G11,sd.Ghatn,2))

#print(sd.Ghatn)