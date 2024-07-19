import numpy as np
#Maps for matrices from Ghat, the first two entries in the touple give the region in G hat, the last entry specifies an overall factor,
#the last list indicates reflections in the axes (s1,s2)
Gm = [[(0, 0, 1), (0, 3, 1), (0,4, 1), (0,7, 1)],
       [(3,0, 1), (3,3, 1), (3,4, 1), (3,7, 1)],
       [(4,0, 1), (4,3, 1), (4,4, 1), (4,7, 1)],
       [(7,0, 1), (7,3, 1), (7,4, 1), (7,7, 1)],
       [0,0]]

Gmtilde = [[(1,1, 1), (1,2, -1), (1,5,1), (1,6,-1)],
       [(2,1, -1), (2,2, 1), (2,5,-1), (2,6,1)],
       [(5,1, 1), (5,2, -1), (5,5,1), (5,6,-1)],
       [(6,1, -1), (6,2, 1), (6,5,-1), (6,6,1)],
       [1,1]]

G_maps = {'Gm': Gm, 'Gmtilde': Gmtilde}

test = -1/2*np.array([
    [1111, 1112, 1212, 1211, -1214, -1213, 1113, 1114, 1115, 1116, 1216, 1215, -1218, -1217, 1117, 1118],
    [1121, 1122, 1222, 1221, -1224, -1223, 1123, 1124, 1125, 1126, 1226, 1225, -1228, -1227, 1127, 1128],
    [2121, 2122, 2222, 2221, -2224, -2223, 2123, 2124, 2125, 2126, 2226, 2225, -2228, -2227, 2127, 2128],
    [2111, 2112, 2212, 2211, -2214, -2213, 2113, 2114, 2115, 2116, 2216, 2215, -2218, -2217, 2117, 2118],
    [-2141, -2142, -2242, -2241, 2244, 2243, -2143, -2144, -2145, -2146, -2246, -2245, 2248, 2247, -2147, -2148],
    [-2131, -2132, -2232, -2231, 2234, 2233, -2133, -2134, -2135, -2136, -2236, -2235, 2238, 2237, -2137, -2138],
    [1131, 1132, 1232, 1231, -1234, -1233, 1133, 1134, 1135, 1136, 1236, 1235, -1238, -1237, 1137, 1138],
    [1141, 1142, 1242, 1241, -1244, -1243, 1143, 1144, 1145, 1146, 1246, 1245, -1248, -1247, 1147, 1148],
    [1151, 1152, 1252, 1251, -1254, -1253, 1153, 1154, 1155, 1156, 1256, 1255, -1258, -1257, 1157, 1158],
    [1161, 1162, 1262, 1261, -1264, -1263, 1163, 1164, 1165, 1166, 1266, 1265, -1268, -1267, 1167, 1168],
    [2161, 2162, 2262, 2261, -2264, -2263, 2163, 2164, 2165, 2166, 2266, 2265, -2268, -2267, 2167, 2168],
    [2151, 2152, 2252, 2251, -2254, -2253, 2153, 2154, 2155, 2156, 2256, 2255, -2258, -2257, 2157, 2158],
    [-2181, -2182, -2282, -2281, 2284, 2283, -2183, -2184, -2185, -2186, -2286, -2285, 2288, 2287, -2187, -2188],
    [-2171, -2172, -2272, -2271, 2274, 2273, -2173, -2174, -2175, -2176, -2276, -2275, 2278, 2277, -2177, -2178],
    [1171, 1172, 1272, 1271, -1274, -1273, 1173, 1174, 1175, 1176, 1276, 1275, -1278, -1277, 1177, 1178],
    [1181, 1182, 1282, 1281, -1284, -1283, 1183, 1184, 1185, 1186, 1286, 1285, -1288, -1287, 1187, 1188],
    ])

#use maps to convert Ghat into G11, G22, etc.
def read_G_from_Ghat(matrix, step):

    matrix_dict = {}

    matrices = []
    map = G_maps['Gm']

    for i in range(len(map)-1):
        row = []
        for j in range(len(map[i])):
            x,y,c = map[i][j]
            new_matrix = np.double(c)*np.array(matrix[step*x:step*(x+1),step*y:step*(y+1)])

            #do any necessary flips
            ax = map[-1]
            if (ax[1]):
                new_matrix = np.flip(new_matrix,axis=1)
            if (ax[0]):
                new_matrix = np.flip(new_matrix,axis=0)

            #put blocks into 2 by 2 form
            row.append(new_matrix)
        matrices.append(row)
    
    return np.block(matrices)

def create_Sigma_hat(matrices, step):
    Sigma_hat = np.zeros((8*step,8*step), dtype=np.double)

    for key in G_maps:
        map = G_maps[key]
        matrix = matrices[key]

        for i in range(len(map)-1):
            for j in range(len(map[i])):
                x,y,c = map[i][j]
                new_matrix = np.array(matrix[step*i:step*(i+1),step*j:step*(j+1)])/np.double(c)

                #do any necessary flips
                ax = map[-1]
                if (ax[1]):
                    new_matrix = np.flip(new_matrix,axis=1)
                if (ax[0]):
                    new_matrix = np.flip(new_matrix,axis=0)

                Sigma_hat[step*x:step*(x+1),step*y:step*(y+1)] = new_matrix

    return Sigma_hat