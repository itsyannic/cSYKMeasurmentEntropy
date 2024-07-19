import numpy as np
#Maps for matrices from Ghat, the first two entries in the touple give the region in G hat, the last entry specifies an overall factor,
#the last list indicates reflections in the axes (s1,s2)
G11 = [[(0, 0, 1), (0, 3, 1), (0,4, 1), (0,7, 1)],
       [(3,0, 1), (3,3, 1), (3,4, 1), (3,7, 1)],
       [(4,0, 1), (4,3, 1), (4,4, 1), (4,7, 1)],
       [(7,0, 1), (7,3, 1), (7,4, 1), (7,7, 1)],
       [0,0]]

G22 = [[(1,1, 1), (1,2, -1), (1,5,1), (1,6,-1)],
       [(2,1, -1), (2,2, 1), (2,5,-1), (2,6,1)],
       [(5,1, 1), (5,2, -1), (5,5,1), (5,6,-1)],
       [(6,1, -1), (6,2, 1), (6,5,-1), (6,6,1)],
       [1,1]]

G12 = [[(0,1,1), (0,2,1), (0,5,1), (0,6, 1)],
       [(3,1,1), (3,2,1), (3,5,1), (3,6, 1)],
       [(4,1,1), (4,2,1), (4,5,1), (4,6, 1)],
       [(7,1,1), (7,2,1), (7,5,1), (7,6, 1)],
       [0,1]]

G21 = [[(1,0,1), (1,3,1), (1,4,1), (1,7,1)],
       [(2,0,1), (2,3,1), (2,4,1), (2,7, 1)],
       [(5,0,1), (5,3,1), (5,4,1), (5,7, 1)],
       [(6,0,1), (6,3,1), (6,4,1), (6,7, 1)],
       [1,0]]

G_maps = {'G11': G11, 'G22': G22, 'G12': G12, 'G21': G21}

#use maps to convert Ghat into G11, G22, etc.
def read_G_from_Ghat(matrix, step):

    matrix_dict = {}

    for key in G_maps:

        map = G_maps[key]

        matrices = []

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

        matrix_dict[key] = np.block(matrices)
    
    return matrix_dict

def create_Sigma_hat(matrices, step):
    Sigma_hat = np.empty((8*step,8*step), dtype=object)

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