import sys
import numpy as np
import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code
from ldpc import mod2
import numpy as np
import itertools
from ldpc import bposd_decoder
from bposd.css import css_code
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from ldpc.codes import rep_code
from bposd.hgp import hgp
from math import comb
from itertools import combinations
# Library for reusing the defect parity generator
def code_dict(name="gross"):
    """
    arg : Name of the code that you want to experiment on. You may add your own codes.
    returns :  ell,m,a1,a2,a3,b1,b2,b3 
    These are to be used by Bicycle Generator to create the code of your liking
    """
    if name == "gross":
        ell,m = 12,6
        a1,a2,a3 = 3,1,2
        b1,b2,b3 = 3,1,2
    if name == "d6":
        #[[72,12,6]]
        ell,m = 6,6
        a1,a2,a3=3,1,2
        b1,b2,b3=3,1,2    
    if name == "d10-90":
        #[[90,8,10]]
        ell,m = 15,3
        a1,a2,a3 = 9,1,2
        b1,b2,b3 = 0,2,7     
    if name == "d10-108":
        # [[108,8,10]]
        ell,m = 9,6
        a1,a2,a3 = 3,1,2
        b1,b2,b3 = 3,1,2   
    if name == "d18":
        # [[288,12,18]]
        ell,m = 12,12
        a1,a2,a3 = 3,2,7
        b1,b2,b3 = 3,1,2
    if name == "d24":
        # [[360,12,24]]
        ell,m = 30,6
        a1,a2,a3 = 9,1,2
        b1,b2,b3 = 3,25,26   
    if name == "d34":
        # [[756,16,34]]
        ell,m = 21,18
        a1,a2,a3 = 3,10,17
        b1,b2,b3 = 5,3,19
    return (ell,m,a1,a2,a3,b1,b2,b3)

def check_unique_unordered_tuples(test_list):
    unique_tuples = {tuple(sorted(t)) for t in test_list}
    return len(unique_tuples)      
def bivariate_parity_generator_bicycle(codeParams):
    """
    input : A 8-tuple with the BB-code's (l,m,a1,a2,a3,b1,b2,b3)
    output : Returns the Css code
    """
    #Extract the parameters
    (ell,m,a1,a2,a3,b1,b2,b3) = codeParams
    # generated lifted code with the specified parameters (Default Values generate the Gross Code)
    # returns the PARITY matrices hx and hz
    n = 2*ell*m
    n2 = ell*m
    I_ell = np.identity(ell,dtype=int)
    I_m = np.identity(m,dtype=int)
    I = np.identity(ell*m,dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
    for i in range(m):
        y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

    # define check matrices
    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2
    AT = np.transpose(A)
    BT = np.transpose(B)
    hx0 = np.hstack((A,B))
    hz0 = np.hstack((BT,AT))
    qcode = css_code(hx0,hz0)
    return qcode     

def connection_matrices(codeParams):
    """
    input : A 8-tuple with the BB-code's (l,m,a1,a2,a3,b1,b2,b3)
    output : Returns the Cyclic Connection Matrices
    """
    #Extract the parameters
    (ell,m,a1,a2,a3,b1,b2,b3) = codeParams
    # generated lifted code with the specified parameters (Default Values generate the Gross Code)
    # returns the PARITY matrices hx and hz
    n = 2*ell*m
    n2 = ell*m
    I_ell = np.identity(ell,dtype=int)
    I_m = np.identity(m,dtype=int)
    I = np.identity(ell*m,dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
    for i in range(m):
        y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

    # define check matrices
    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2
    AT = np.transpose(A)
    BT = np.transpose(B)
    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    return A1,A2,A3,B1,B2,B3
def damage_qubit(Q,turnOffQubits=[0],symmetry=False,alterning = False,type='data'):
    # Turns off the qubits at specified positions and recalculates the parity matrix
    if type == 'data':
        hxq = Q.hx.copy()
        hzq = Q.hz.copy()
        Q_New = css_code(hxq,hzq)
        Flip = False
        print("Flip is ",False)
        #np.random.shuffle(turnOffQubits)
        for defects in turnOffQubits:
            #print("Defect Deteced at ",defects)
            # Find all rows where the turnOffQubits are high.To figure out the connectivity
            # X-Z syndromes
            ALL_THREE = False
            TWO_AT_A_TIME = True
            SPELL_IT_OUT = False #Read out the defective stabilizers
            HIGH = 1
            LOW = 1
            hx = Q_New.hx
            hz = Q_New.hz
            broken_rows_x = hx[hx[:,defects] == HIGH]

            broken_rows_x_DEBUG = broken_rows_x.copy()
            broken_rows_z = hz[hz[:,defects] == HIGH]
            broken_rows_z_DEBUG = broken_rows_z.copy()
            print("SANITY CHECK ",np.array_equal(broken_rows_x_DEBUG,broken_rows_z_DEBUG))
            if SPELL_IT_OUT:
                # Read out the browen_rows
                defectNum,_ = broken_rows_x_DEBUG.shape
                print("Defective Positions in X/Z")
                for i in range(defectNum):
                    affectedRowx = broken_rows_x_DEBUG[i]
                    affectedRowz = broken_rows_z_DEBUG[i]
                    stab_X_read = np.where(affectedRowx == HIGH)
                    stab_Z_read = np.where(affectedRowz == HIGH)
                    print("StabX is ",stab_X_read)
                    print("StabZ is ",stab_Z_read)
            rows_to_be_deleted_x = np.where(hx[:,defects]==HIGH)[0]
            rows_to_be_deleted_z = np.where(hz[:,defects] == HIGH)[0]
            # This is importnat. To understand relative orientation
            print("x deleted rows should be ",rows_to_be_deleted_x)
            print("z deleted rows should be ",rows_to_be_deleted_z)
            RAND = False
            if RAND:
                np.random.shuffle(rows_to_be_deleted_x)
                np.random.shuffle(rows_to_be_deleted_z)
            # Save the first column because we will replace it with the superstabilizer
            hx_old = hx.copy()
            hz_old = hz.copy()
            if TWO_AT_A_TIME:
                # HARD CODE RN [0,1,2]
                # AB
                if symmetry:
                    hx[rows_to_be_deleted_x[0]] = (hx_old[rows_to_be_deleted_x[1]]+hx_old[rows_to_be_deleted_x[2]])%2
                if Flip:
                    hx[rows_to_be_deleted_x[0]] = (hx_old[rows_to_be_deleted_x[1]]+hx_old[rows_to_be_deleted_x[2]])%2
                else:    
                    hz[rows_to_be_deleted_z[0]] = (hz_old[rows_to_be_deleted_z[1]]+hz_old[rows_to_be_deleted_z[2]])%2
                # BC
                if symmetry:
                    hx[rows_to_be_deleted_x[1]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[2]])%2
                if Flip:
                    hx[rows_to_be_deleted_x[1]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[2]])%2
                else:
                    hz[rows_to_be_deleted_z[1]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[2]])%2
                # CA
                if symmetry:
                    hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
                if Flip:
                    hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
                else:
                    hz[rows_to_be_deleted_z[2]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[1]])%2
                #hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
                #hz[rows_to_be_deleted_z[2]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[1]])%2
                DB = True
                if  DB :
                    if Flip:
                        hx[rows_to_be_deleted_x[2]] = 0* hx[rows_to_be_deleted_x[2]]
                    else:
                        hz[rows_to_be_deleted_z[2]] = 0* hz[rows_to_be_deleted_z[2]]
                    if symmetry:
                        hx[rows_to_be_deleted_x[2]] = 0* hx[rows_to_be_deleted_x[2]]
                # Delete the extra row
                #hx = np.delete(hx,rows_to_be_deleted_x[2],axis=0)
                #hz = np.delete(hz,rows_to_be_deleted_z[2],axis=0)
            replace_x = rows_to_be_deleted_x[0]
            replace_z = rows_to_be_deleted_z[0]
            print("To be deleted ",rows_to_be_deleted_x)
           
            if alterning:
                Flip = not Flip
        # Deleted at the end because the numbering goes out of sync
        #Experiments
        #hx[:,turnOffQubits] = 0
        #hz[:,turnOffQubits] = 0
        hx = np.delete(hx,turnOffQubits,axis=1)
        hz = np.delete(hz,turnOffQubits,axis=1)
        Q_New = css_code(hx,hz)
        Q_New.test()
        return Q_New
    else:
        # Destroy Parity Measurement Qubits
        #By default kill z qubits ,thats what we want
        hz = Q.hz.copy()
        hx = Q.hx.copy()
        #Delete bad columns in z 
        hz = np.delete(hz,turnOffQubits,axis=0)
        Q_New = css_code(hx,hz)
        return Q_New
