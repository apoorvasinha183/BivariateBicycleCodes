# We experiment killing parities here
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
# Create a method to return dictionaries based retrieval of code and code
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

# The main method would MC sample upto 3 errors for each code 
if __name__ == "__main__":
    
    #codes = ["d6","d10-90","d10-108","gross","d18","d24","d34"]
    codes =["d34"]
    #codes = ["d6","d10-90","d10-108","gross","d18","d24"]
    nKill = 3
    for code in codes:
        # Get the code
        Q = bivariate_parity_generator_bicycle(code_dict(name=code))
        print("Error Resistance Check on Code for ",nKill, " parity qubits")
        Q.test()
        #print("Error resitsance on ")
        kmax = Q.K +1 
        nTrials = 100000
        #nKill = 1
        # Choose nKill parity to kill .The first WLOG can be 0 . For a first Order check sample randomly
        # Small Code
        nChoose = int(Q.N/2) 
        #signature = []
        flip = 0
        bad_places = []
        trail = []
        #0 is fixed so we see how many unique combos we need first
        exhaustiveTrialNum = comb(nChoose-1,nKill-1)
        print("There are a total of ",exhaustiveTrialNum," unique error combinations.")
        EXHAUST = True
        # Explicitly find the combinations and iterate
        rangeToTurnOff = np.arange(1,nChoose)
        all_combos = list(combinations(rangeToTurnOff.tolist(),nKill-1))
        for i in tqdm(range(nTrials)):
            signature =[]
            if EXHAUST:
                turnOfflines = np.array(all_combos[i])
            else:
                turnOfflines = np.random.choice(np.arange(1,nChoose),size=nKill-1,replace= False)
            trail.append(tuple(turnOfflines.tolist()))
            turnOfflines = np.append(turnOfflines,0)
            hx = Q.hx.copy()
            hz = Q.hz.copy()
            for lines in turnOfflines:
                # Do a 50-50 coin toss 
                choose = np.random.rand()
                if choose > 1:
                    hx[lines,:] = 0 * hx[lines,:]
                    signature.append('x')
                else:
                    hz[lines,:] = 0 * hz[lines,:]	
                    signature.append('z')
            qCodeNew = css_code(hx,hz)
            k_sample = qCodeNew.K
            if k_sample >= kmax:
                flip += 1
                #print("Present max is ",k_sample)
                #print("The bad configurqation is ",turnOfflines)
                #print("Error signature is ",signature)
                signature = []
                bad_places.append(tuple(turnOfflines.tolist()))
                kmax = k_sample
            #Check if the number of trials is that much that we actually sampled the full size 
            if check_unique_unordered_tuples(trail) >= exhaustiveTrialNum:
                print("Every sample obtained.")
                break    
        print("Maximum qubits is ",kmax)	
        print("Total number of flips in 10000 : ",flip)		
        print("code test")
        #Q.test()
        #bad_places = np.unique(bad_places)
        print(" Bad turn off locations are ",check_unique_unordered_tuples(bad_places))
        print("We tried the following number of unique combinations ",check_unique_unordered_tuples(trail))
