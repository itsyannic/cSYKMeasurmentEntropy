import numpy
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(1,1,1,0,4)

matrices = fields.convert_map_to_matrix(fields.test,2)

maps = [fields.G11, fields.G22, fields.G12, fields.G21]

test2 = fields.invert_maps(matrices,2)

print(fields.test-test2)