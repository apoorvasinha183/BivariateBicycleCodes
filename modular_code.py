import numpy as np
import itertools
from ldpc import bposd_decoder
from bposd.css import css_code
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse import hstack

###### SAME AS BRAVYI's CODE ######
# Takes as input a binary square matrix A
# Returns the rank of A over the binary field F_2
def rank2(A):
    rows,n = A.shape
    X = np.identity(n,dtype=int)

    for i in range(rows):
        y = np.dot(A[i,:], X) % 2
        not_y = (y + 1) % 2
        good = X[:,np.nonzero(not_y)]
        good = good[:,0,:]
        bad = X[:, np.nonzero(y)]
        bad = bad[:,0,:]
        if bad.shape[1]>0 :
            bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]
###### SAME AS BRAVYI's CODE ######

### RE-ROGANIZED MODULAR CODE ###
