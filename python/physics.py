import numpy as np
import fields
from SchwingerDyson import SchwingerDyson

def on_shell_action_den(SD_Object: SchwingerDyson):

    m = SD_Object.m
    term1 = -m/2*(np.log(np.linalg.det(np.linalg.inv(SD_Object.Ghat_d_free_inverse@SD_Object.Ghatd))) + 2*np.log(2,dtype=np.double)) 
    term2 = -(1-m)*(np.log(np.linalg.det(np.linalg.inv(SD_Object.G33_d_free_inverse@SD_Object.G33d))) + 2*np.log(2,dtype=np.double))
    Gdij = fields.read_G_from_Ghat(SD_Object.Ghatd, int(SD_Object.discretization/2))
    brace = -m/2*Gdij['G11'] + m/2*Gdij['G22'] - m/2*Gdij['G12'] + m/2*Gdij['G21'] + (1-m)*SD_Object.G33d
    Gsqr = np.power(brace,SD_Object.q/2)
    term3 = -SD_Object.Jsqr*(1/SD_Object.q - 1)*np.trace(Gsqr@Gsqr)

    return term1 + term2 + term3

def on_shell_action_num(SD_Object: SchwingerDyson):

    m = SD_Object.m
    term1 = -m/2*(np.log(np.linalg.det(np.linalg.inv(SD_Object.Ghat_n_free_inverse@SD_Object.Ghatn))) + 2*np.log(2,dtype=np.double)) 
    term2 = -(1-m)*(np.log(np.linalg.det(np.linalg.inv(SD_Object.G33_n_free_inverse@SD_Object.G33n))) + np.log(2,dtype=np.double))
    Gdij = fields.read_G_from_Ghat(SD_Object.Ghatn, int(SD_Object.discretization/2))
    brace = -m/2*Gdij['G11'] + m/2*Gdij['G22'] - m/2*Gdij['G12'] + m/2*Gdij['G21'] + (1-m)*SD_Object.G33n
    Gsqr = np.power(brace,SD_Object.q/2)
    term3 = -SD_Object.Jsqr*(1/SD_Object.q - 1)*np.trace(Gsqr@Gsqr)

    return term1 + term2 + term3

def charge(SD_Object: SchwingerDyson):

    return SD_Object.m/2 #+f(G33)

def renyi2(Iden, Inum):

    return Inum - Iden

def results(SD_Object: SchwingerDyson):

    Inum = on_shell_action_num(SD_Object)
    Iden = on_shell_action_den(SD_Object)
    I2 = renyi2(Iden, Inum)
    Q = charge(SD_Object)

    return {'m': SD_Object.m, 'Inum': Inum, 'Iden': Iden, 'renyi2': I2, 'charge': Q}