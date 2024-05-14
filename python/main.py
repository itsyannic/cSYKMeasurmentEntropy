import numpy
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(1,1,1,0,4)

print(fields.convert_map_to_matrix(fields.G22,fields.test,2))

#print(sd.Ghatn)