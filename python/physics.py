import numpy as np
import fields
from SchwingerDyson import SchwingerDyson

def on_shell_action_den(SD_Object: SchwingerDyson):

    m = SD_Object.m
    arg = np.linalg.det(np.linalg.inv(SD_Object.G11_d_free_inverse@SD_Object.G11d))
    term1 = -m*(np.log(arg) + np.log(2,dtype=np.double) ) 
    term2 = -(1-m)*(np.log(np.linalg.det(np.linalg.inv(SD_Object.G33_d_free_inverse@SD_Object.G33d))) + 2*np.log(2,dtype=np.double))
    brace = m*SD_Object.G11d + (1-m)*SD_Object.G33d
    Gsqr = np.power(brace,SD_Object.q/2)
    term3 = -SD_Object.Jsqr*(1/SD_Object.q - 1)*np.trace(Gsqr@Gsqr)/(SD_Object.discretization*2)**2

    return term1 + term2 + term3

def on_shell_action_num(SD_Object: SchwingerDyson):

    m = SD_Object.m
    arg = np.linalg.det(np.linalg.inv(SD_Object.G11_n_free_inverse@SD_Object.G11n))
    term1 = -m*(np.log(arg) ) 
    term2 = -(1-m)*(np.log(np.linalg.det(np.linalg.inv(SD_Object.G33_n_free_inverse@SD_Object.G33n))) + np.log(2,dtype=np.double))
    brace = m*SD_Object.G11n + (1-m)*SD_Object.G33n
    Gsqr = np.power(brace,SD_Object.q/2)
    term3 = -SD_Object.Jsqr*(1/SD_Object.q - 1)*np.trace(Gsqr@Gsqr)/(SD_Object.discretization*2)**2

    return term1 + term2 + term3

def trG33(SD_Object: SchwingerDyson):
    return np.trace(SD_Object.G33n)/len(SD_Object.G33n[0])

def charge(SD_Object: SchwingerDyson, Iden, trG):
    #print((1-SD_Object.m)*np.trace(SD_Object.G33n))
    #return -SD_Object.m/2
    return SD_Object.m/2  - (1-SD_Object.m)*trG #?

def renyi2(Iden, Inum):

    return Inum - Iden

def results(SD_Object: SchwingerDyson):

    Inum = on_shell_action_num(SD_Object)
    Iden = on_shell_action_den(SD_Object)
    trG = trG33(SD_Object)
    I2 = renyi2(Iden, Inum)
    Q = charge(SD_Object,Iden,trG)

    return {'m': SD_Object.m, 'Inum': Inum, 'Iden': Iden, 'renyi2': I2, 'charge': Q, 'trG33': trG}