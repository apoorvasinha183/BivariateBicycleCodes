# Stim setups for the circuits 
# This creates the usual naive syndrome extraction circuit for the BB-codes. 
import pathlib

from typing import Dict
from sinter import Decoder, CompiledDecoder
import numpy as np
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
import stim

from stimbposd.bp_osd import BPOSD
#TODO: Use Gross Code Paper data
from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
    DEFAULT_OSD_ORDER,
    DEFAULT_OSD_METHOD,
)


## --  Methods to create Parity Check Matrices (INIT)  -- ##
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
## --  Methods to create Parity Check Matrices (END)  -- ##

## -- METHODS TO GLUE THE STIM CIRCUIT (INIT) --##
def parity_to_stim(qCode,perr=0.001,rounds= 12):
    """
    inputs: Quantum Code  from Roffee's code
    output : Stim Circuit(also the .stim file is generated which is explicitly simulated)"""
    # Number of Qubits 
    N = qCode.N
    # Number of X checks
    Nx = qCode.hx.shape[0]
    # Number of Z checks 
    Nz = qCode.hz.shape[0]
    #TODO : Do this in an aesthetic manner with Qubit_coords 
    









## -- METHODS TO GLUE THE STIM CIRCUIT (END) --##