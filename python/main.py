import numpy
import SchwingerDyson
import fields

sd = SchwingerDyson.SchwingerDyson(1,1,1,0,4)

G11 = fields.convert_map_to_matrix(fields.G11, fields.test,2)
G22 = fields.convert_map_to_matrix(fields.G22, fields.test,2)
G12 = fields.convert_map_to_matrix(fields.G12, fields.test,2)
G21 = fields.convert_map_to_matrix(fields.G21, fields.test,2)

matrices = [G11,G22,G12,G21]
maps = [fields.G11, fields.G22, fields.G12, fields.G21]

test2 = fields.invert_maps(maps,matrices,2)

print(fields.test-test2)