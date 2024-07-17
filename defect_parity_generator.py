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
def ncycles(name="gross"):
    """
    arg : Name of the code that you want to experiment on. You may add your own codes.
    returns :  ell,m,a1,a2,a3,b1,b2,b3 
    These are to be used by Bicycle Generator to create the code of your liking
    """
    rounds = 1
    if name == "gross":
        rounds = 12
    if name == "d6":
        #[[72,12,6]]
        rounds = 6  
    if name == "d10-90":
        #[[90,8,10]]
        rounds = 10   
    if name == "d10-108":
        # [[108,8,10]]
        rounds = 10
    if name == "d18":
        # [[288,12,18]]
        rounds = 18
    if name == "d24":
        # [[360,12,24]]
        rounds = 18  
    if name == "d34":
        # [[756,16,34]]
        rounds = 34
    return rounds
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
def order_extract(array_original,array_new):
    order_correct = []
    for i in range(3):
        elem = array_original[i]
        for j in range(3): # Don't do this in an interview!
            if elem == array_new[j]:
                order_correct.append(j)
    return order_correct


def damage_qubit(Q,turnOffQubits=[0],symmetry=False,alterning = False,type='data',flip = False,test=False,order=False):
    # Turns off the qubits at specified positions and recalculates the parity matrix
    ##print("Attempt to delete qubit ",turnOffQubits)
    fixed = False
    RAND = False
    # To track that we don't put anything on a line
    row_x_prev = []
    row_z_prev = []
    min_wtt = 200
    max_wt = 0
    while(not fixed):
        cont = False
        if type == 'data':
            print("Turn off qubits ",turnOffQubits)
            print("ZX repair is ",alterning)
            hxq = Q.hx.copy()
            hzq = Q.hz.copy()
            Q_New = css_code(hxq,hzq)
            Flip = flip
            ##print("Flip is ",Flip)
            ##print("Parity break is ",symmetry)
            hx = Q_New.hx
            hz = Q_New.hz
            #np.random.shuffle(turnOffQubits)
            order_x = []
            order_z = []
            offset = 0
            anchors= []
            #anchor_z = []
            style = 'z'
            Xbad = []
            Zbad = []
            hx_o = hx.copy()
            hz_o = hz.copy()
            row_x_prev_loop = row_x_prev.copy()
            row_z_prev_loop = row_z_prev.copy()
            for defects in turnOffQubits:
                if Flip:
                    style = 'x'
                else:
                    style = 'z'    
                ##print("Defect Deteced at ",defects)
                # Find all rows where the turnOffQubits are high.To figure out the connectivity
                # X-Z syndromes
                ALL_THREE = False
                TWO_AT_A_TIME = True
                SPELL_IT_OUT = False #Read out the defective stabilizers
                HIGH = 1
                LOW = 1
                
                broken_rows_x = hx_o[hx_o[:,defects] == HIGH]

                broken_rows_x_DEBUG = broken_rows_x.copy()
                broken_rows_z = hz_o[hz_o[:,defects] == HIGH]
                broken_rows_z_DEBUG = broken_rows_z.copy()
                #print("SANITY CHECK ",np.array_equal(broken_rows_x_DEBUG,broken_rows_z_DEBUG))
                if SPELL_IT_OUT:
                    # Read out the browen_rows
                    defectNum,_ = broken_rows_x_DEBUG.shape
                    #print("Defective Positions in X/Z")
                    for i in range(defectNum):
                        affectedRowx = broken_rows_x_DEBUG[i]
                        affectedRowz = broken_rows_z_DEBUG[i]
                        stab_X_read = np.where(affectedRowx == HIGH)
                        stab_Z_read = np.where(affectedRowz == HIGH)
                        #print("StabX is ",stab_X_read)
                        #print("StabZ is ",stab_Z_read)
                rows_to_be_deleted_x = np.where(hx[:,defects]==HIGH)[0]
                rows_to_be_deleted_z = np.where(hz[:,defects] == HIGH)[0]
                # This is importnat. To understand relative orientation
                rows_to_be_deleted_x_original = np.where(hx_o[:,defects]==HIGH)[0]
                rows_to_be_deleted_z_original = np.where(hz_o[:,defects] == HIGH)[0]
                print("x deleted rows should be ",rows_to_be_deleted_x_original)
                print("z deleted rows should be ",rows_to_be_deleted_z_original)
                print("x deleted rows cumulative should be ",rows_to_be_deleted_x_original)
                print("z deleted rows cumulative should be ",rows_to_be_deleted_z_original)
                #rows_to_be_deleted_x_original = rows_to_be_deleted_x.copy()
                #rows_to_be_deleted_z_original = rows_to_be_deleted_z.copy()
                # Check for overlap with the previous sets
                row_z_prev_loop = rows_to_be_deleted_z.copy()
                order_x_new = [0,1,2]
                order_z_new = [0,1,2]
                #RAND = False
                if RAND:
                    np.random.shuffle(rows_to_be_deleted_x)
                    np.random.shuffle(rows_to_be_deleted_z)
                    print("shuffled rowx ",rows_to_be_deleted_x)
                    print("shuffled rowz ",rows_to_be_deleted_z)
                    #order_x_new = order_extract(rows_to_be_deleted_x_original,rows_to_be_deleted_x)
                    #order_z_new = order_extract(rows_to_be_deleted_z_original,rows_to_be_deleted_z)
                # Save the first column because we will replace it with the superstabilizer
                if style == 'z':
                    anchors.append(('Zcheck',rows_to_be_deleted_z[2]))
                else:
                    anchors.append(('Xcheck',rows_to_be_deleted_x[2]))
                # Explicitly send the expected order to the circuit
                if alterning:
                    if Flip:
                        for i in range(3):
                            chkName = ('Xcheck',rows_to_be_deleted_x[i])
                            if i < 2:
                                if chkName in anchors:
                                    cont = True
                            if i ==2:
                                if chkName in Xbad:
                                    cont = True
                            Xbad.append(chkName)
                    else:
                        for i in range(3):
                            chkName = ('Zcheck',rows_to_be_deleted_z[i])
                            if i < 2:
                                if chkName in anchors:
                                    cont = True
                            if i==2:
                                if chkName in Zbad:
                                    cont = True

                            Zbad.append(chkName)
                else:
                    for i in range(3):
                        chkNameX = ('Xcheck',rows_to_be_deleted_x[i])
                        chkNameZ = ('Zcheck',rows_to_be_deleted_z[i])
                        if i<2:
                            if chkNameX in anchors:
                                    cont = True
                            if chkNameZ in anchors:
                                    cont = True
                        if i==2:
                            if chkNameX in Xbad:
                                    cont = True
                            if chkNameZ in Zbad:
                                    cont = True

                        Xbad.append(chkNameX)
                        Zbad.append(chkNameZ)


                #print("order_z_new ",order_z_new)
                
                order_x += order_x_new
                order_z += order_z_new
                hx_old = hx.copy()
                hz_old = hz.copy()
                if TWO_AT_A_TIME:
                    # HARD CODE RN [0,1,2]
                    # AB
                    if symmetry or Flip:
                        hx[rows_to_be_deleted_x[0]] = (hx_old[rows_to_be_deleted_x[1]]+hx_old[rows_to_be_deleted_x[2]])%2
                    if not Flip:      
                        hz[rows_to_be_deleted_z[0]] = (hz_old[rows_to_be_deleted_z[1]]+hz_old[rows_to_be_deleted_z[2]])%2
                    # BC
                    if symmetry or Flip:
                        hx[rows_to_be_deleted_x[1]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[2]])%2
                    if not Flip:
                        hz[rows_to_be_deleted_z[1]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[2]])%2
                    # CA
                    if symmetry or Flip:
                        hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
                    if not Flip:
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
                #print("To be deleted ",rows_to_be_deleted_x)
                offset += 3
                if alterning:
                    Flip = not Flip
            # Deleted at the end because the numbering goes out of sync
            #Experiments
            #hx[:,turnOffQubits] = 0
            #hz[:,turnOffQubits] = 0
            if cont:
                print("Unsilenced line")
                RAND = True
                continue
            hx = np.delete(hx,turnOffQubits,axis=1)
            hz = np.delete(hz,turnOffQubits,axis=1)
            Q_New = css_code(hx,hz)
            if test:
                valid = Q_New.test()
                if valid:
                    print("THIS CODE IS VALID")
                    fixed = True
                else:
                    print(f"Discard.Retry.DeadParityLocations:{turnOffQubits}")
                    RAND = True
                    fixed = False
                    continue
            if order:
                print(f"Shuffled Order-x is {order_x}")
                print(f"Shuffled Order-z is {order_z}")
                print(f"anchors are {anchors}")
                print(f"Xbad:{Xbad}Zbad:{Zbad}")
                for rows in hz:
                    wt = np.sum(rows)
                    #print(f"Weight is {wt}")
                    if wt < min_wtt:
                        min_wtt = wt
                    if wt > max_wt:
                        max_wt = wt
                print(f"Minimum weight is {min_wtt}:Maximum weight:{max_wt}")
                Q_New.test()
                return Q_New,anchors,Xbad,Zbad
            else:
                print(f"anchors are {anchors}")
                for rows in hz:
                    wt = np.sum(rows)
                    #print(f"Weight is {wt}")
                    if wt < min_wtt:
                        min_wtt = wt
                    if wt > max_wt:
                        max_wt = wt
                print(f"Minimum weight is {min_wtt}:Maximum weight:{max_wt}")
                Q_New.test()
                return Q_New
        else:
            # Destroy Parity Measurement Qubits
            #By default kill z qubits ,thats what we want
            hz = Q.hz.copy()
            hx = Q.hx.copy()
            #Delete bad columns in z 
            hz = np.delete(hz,turnOffQubits,axis=0)
            min_wtt = 200
            for rows in hz:
                wt = np.sum(rows)
                #print(f"Weight is {wt}")
                if wt < min_wtt:
                    min_wtt = wt
            print(f"Minimum weight is {min_wtt}")
            Q_New = css_code(hx,hz)
            return Q_New
def damage_qubit_v2(Q,turnOffQubits=[0],symmetry=False,alterning = False,type='data',flip = False,test=False,order=False):
    print("Turn off qubits ",turnOffQubits)
    #print("ZX repair is ",alterning)
    
    distinct_wt_z = []
    distinct_wt_x = []
    HIGH = 1
    LOW = 0
    hxq = Q.hx.copy()
    hzq = Q.hz.copy()
    Q_New = css_code(hxq,hzq)
    Flip = flip
    ##print("Flip is ",Flip)
    ##print("Parity break is ",symmetry)
    hx = Q_New.hx
    hz = Q_New.hz
    print("Weird distant qubits ",np.where(hz[0]==HIGH))
    print("Weird distant qubits ",np.where(hz[36]==HIGH))
    #np.random.shuffle(turnOffQubits)
    order_x = []
    order_z = []
    offset = 0
    anchors= []
    #anchor_z = []
    style = 'z'
    Xbad = []
    Zbad = []
    hx_o = hx.copy()
    hz_o = hz.copy()
    for defects in turnOffQubits:
        broken_rows_x = hx_o[hx_o[:,defects] == HIGH]
        broken_rows_z = hz_o[hz_o[:,defects] == HIGH]
        X_conn = []
        Z_conn = []
        rows_to_be_deleted_x = np.where(hx[:,defects]==HIGH)[0]
        rows_to_be_deleted_z = np.where(hz[:,defects] == HIGH)[0]
        print("Rows-z to be deleted are ",rows_to_be_deleted_x)
        # Look at the Z broken check and its constitutent qubits
        for zrows in broken_rows_z:
            zbroken = np.where(zrows == HIGH)[0]
            zbroken.astype(int)
            print(f"zbroken :{zbroken}")
            zbroken_filtered = zbroken[zbroken != defects]
            Z_conn += list(zbroken_filtered)
        #Look at the X broken checks and their constituent qubits
        for xrows in broken_rows_x:
            xbroken = np.where(xrows == HIGH)[0]
            xbroken_filtered = xbroken[xbroken != defects]
            X_conn += list(xbroken_filtered)
        print(f"Xconn :{X_conn};Zconn:{Z_conn}")
        common = list(set(X_conn) & set(Z_conn))
        print(f"Common Data qubit Connections :{common}")
        #Unhook the common qubits
        # Start with only-Z
        if not Flip:
            for rows in rows_to_be_deleted_z:
                row_chk = hz[rows].copy()
                row_chk[common] = LOW
                hz[rows] = LOW
        if Flip or symmetry:
            for rows in rows_to_be_deleted_x:
                row_chk = hx[rows].copy()
                row_chk[common] = LOW
                hx[rows] = LOW
        if alterning:
            Flip = not Flip

        
    hx = np.delete(hx,turnOffQubits,axis=1)
    hz = np.delete(hz,turnOffQubits,axis=1)
    min_wtt = 200
    for rows in hz:
        wt = np.sum(rows)
        if wt not in distinct_wt_z:
            distinct_wt_z.append(wt)
        #print(f"Weight is {wt}")
        if wt < min_wtt:
            min_wtt = wt
    for rows in hx:
        wt = np.sum(rows)
        if wt not in distinct_wt_x:
            distinct_wt_x.append(wt)
    print(f"Minimum weight is {min_wtt}")
    print("Distinct weights-Z are ",distinct_wt_z)
    print("Distinct weights-X are ",distinct_wt_x)
    Q_New = css_code(hx,hz)
    Q_New.test()
        
    return Q_New



if __name__ == "__main__":
	codeName = code_dict(name="gross")
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_One = damage_qubit_v2(bikeCode,turnOffQubits=[0,5])
	
	