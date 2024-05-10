import numpy as np
#Maps for matrices from Ghat, the first two entries in the touple give the region in G hat, the last entry specifies an overall factor,
#the last list indicates reflections in the axes (s1,s2)
G11 = [[(0, 0, -1/2), (0, 3, -1/2), (0,4, -1/2), (0,7, -1/2)],
       [(3,0, -1/2), (3,3, -1/2), (3,4, -1/2), (3,7, -1/2)],
       [(4,0, -1/2), (4,3, -1/2), (4,4, -1/2), (4,7, -1/2)],
       [(7,0, -1/2), (7,3, -1/2), (7,4, -1/2), (7,7, -1/2)],
       [0,0]]

G22 = [[(1,1, -1/2), (1,2, 1/2), (1,5,-1/2), (1,6,1/2)],
       [(2,1, 1/2), (2,2, -1/2), (2,5,1/2), (2,6,-1/2)],
       [(5,1, -1/2), (5,2, 1/2), (5,5,-1/2), (5,6,1/2)],
       [(6,1, 1/2), (6,2, -1/2), (6,5,1/2), (6,6,-1/2)],
       [1,1]]

G12 = [[(0,1,-1/2), (0,2,1/2), (0,5,-1/2), (0,6, 1/2)],
       [(3,1,-1/2), (3,2,1/2), (3,5,-1/2), (3,6, 1/2)],
       [(4,1,-1/2), (4,2,1/2), (4,5,-1/2), (4,6, 1/2)],
       [(7,1,-1/2), (7,2,1/2), (7,5,-1/2), (7,6, 1/2)],
       [0,1]]

G21 = [[(1,0,-1/2), (1,3,-1/2), (1,4,-1/2), (1,7,-1/2)],
       [(2,0,1/2), (2,3,1/2), (2,4,1/2), (2,7, 1/2)],
       [(5,0,-1/2), (5,3,-1/2), (5,4,-1/2), (5,7, -1/2)],
       [(6,0,1/2), (6,3,1/2), (6,4,1/2), (6,7, 1/2)],
       [1,0]]

#use maps to convert Ghat into G11, G22, etc.
def convert_map_to_matrix(map, matrix, step):

    matrices = []

    for i in range(len(map)-1):
        inner_matrix = []
        for j in range(len(map[i])):
            x,y,c = map[i][j]
            new_matrix = c*np.array([matrix[x+k][y:y+step] for k in range(step)])

            #do any necessary flips
            ax = map[-1]
            if (ax[0]):
                new_matrix = np.flip(new_matrix,axis=1)
            if (ax[1]):
                new_matrix = np.flip(new_matrix,axis=0)

            #put blocks into 2 by 2 form
            inner_matrix.append(new_matrix)
        matrices.append(inner_matrix)

    return np.block(matrices)
