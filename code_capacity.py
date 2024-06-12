# Code Capacity Simulations
# Simple Random X and Z errors of the Physical Qubit length are performed
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
def generateError(pError,Q):
    # Q is expected to be a CSS Code object as created by Roffee's Code
    # Returns X and Z syndromes
    Nsyn = Q.N   # This is not the syndrome but an error operator. Syndrome is created after the Hmatrix is multiplied.
    XErr = np.zeros(Nsyn)
    ZErr = np.zeros(Nsyn)
    for i in range(Nsyn):
        shotX = np.random.rand()
        if shotX < pError:
            XErr[i] = 1
        #Toss the coin again since you do not want correlated X-Z errors
        shotZ = np.random.rand()
        if shotZ < pError:
            ZErr[i] = 1
    # Integer woes
    XErr =XErr.astype(int)
    ZErr = ZErr.astype(int)
    return XErr,ZErr


def bivariate_parity_generator_bicycle(ell = 12,m=6,a1=3,a2=1,a3=2,b1=3,b2=1,b3=2):
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
    return hx0,hz0

def damage_qubit(Q,turnOffQubits=[0],symmetry=False):
    # Turns off the qubits at specified positions and recalculates the parity matrix
    # TODO: Extend it to multpile qubits
    hxq = Q.hx
    hzq = Q.hz
    Q_New = css_code(hxq,hzq)
    for defects in turnOffQubits:
        print("Defect Deteced")
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
        # Save the first column because we will replace it with the superstabilizer
        hx_old = hx.copy()
        hz_old = hz.copy()
        if TWO_AT_A_TIME:
            # HARD CODE RN [0,1,2]
            # AB
            if symmetry:
                hx[rows_to_be_deleted_x[0]] = (hx_old[rows_to_be_deleted_x[1]]+hx_old[rows_to_be_deleted_x[2]])%2
            hz[rows_to_be_deleted_z[0]] = (hz_old[rows_to_be_deleted_z[1]]+hz_old[rows_to_be_deleted_z[2]])%2
            # BC
            if symmetry:
                hx[rows_to_be_deleted_x[1]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[2]])%2
            hz[rows_to_be_deleted_z[1]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[2]])%2
            # CA
            if symmetry:
                hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
            #hz[rows_to_be_deleted_z[2]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[1]])%2
            hz[rows_to_be_deleted_z[2]] = 0* hz[rows_to_be_deleted_z[2]]
            if symmetry:
                hx[rows_to_be_deleted_x[2]] = 0* hx[rows_to_be_deleted_x[2]]
            # Delete the extra row
            #hx = np.delete(hx,rows_to_be_deleted_x[2],axis=0)
            #hz = np.delete(hz,rows_to_be_deleted_z[2],axis=0)
        replace_x = rows_to_be_deleted_x[0]
        replace_z = rows_to_be_deleted_z[0]
        print("To be deleted ",rows_to_be_deleted_x)
        hxBAD = hx[rows_to_be_deleted_x]
        hzBAD = hz[rows_to_be_deleted_z]
        ##### GAUGE APPROACH ##########
        #hx = np.delete(hx,rows_to_be_deleted_x,axis=0)
        #hz = np.delete(hz,rows_to_be_deleted_z,axis=0)

        ##### GAUGE APPROACH ##########
        
        #for rows in rows_to_be_deleted_x:
        #		hx = np.delete(hx,rows,axis = 0 )
        #hx = np.delete(hx,rows_to_be_deleted_x[1:],axis = 0 )
        #hz = np.delete(hz,rows_to_be_deleted_z[1:],axis = 0 )
        #hx[:,defects] =LOW
        #hz[:,defects] = LOW
        #hx[replace_x] = np.sum(broken_rows_x,axis=0) %2
        #hz[replace_z] = np.sum(broken_rows_z,axis=0) %2
        # Now kill the qubit
        hx = np.delete(hx,defects,axis=1)
        hz = np.delete(hz,defects,axis=1)
        hxBAD = np.delete(hxBAD,defects,axis=1)
        hzBAD = np.delete(hzBAD,defects,axis=1)
        #hx[:,defects] = LOW
        #hz[:,defects] = LOW

        #hx = hx[:,1:] 
        #hz = hz[:,1:]
    Q_New = css_code(hx,hz)
    return Q_New

def bposd_decode(qCode,type='css',perr=0.01,nTrials = 10000):
    #Runs Roffee's BP-OSD Decoder to infer the syndrome
    HdecX = qCode.hz
    channel_probsX = perr
    K = qCode.K
    # Declare the Decoder(Re-using what the Gross Code uses)
    my_bp_method = "ms"
    my_max_iter = 10000
    my_osd_method = "osd_cs"
    my_osd_order = 7
    my_ms_scaling_factor = 0
    bpdX=bposd_decoder(
    HdecX,#the parity check matrix
    error_rate=channel_probsX, #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=my_max_iter, #the maximum number of iterations for BP)
    bp_method=my_bp_method,
    ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method=my_osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=my_osd_order #the osd search depth
    )
    # My simulation strategy mirrors the original one by Roffee (they are the same as what Gross Code authors do in spirit)
    # Sample a X/Z error -> Extract its syndrome using (Hz/Hx)(the order respec tive) sX/sZ
    # sX/sZ is decoded by using bpdX/bpdZ
    # inferred error is XOR'd with the actual error string to get residual error res_X/Z
    # This is a logical error if this doesn't commute with a lz/lX operator
    error = 0
    minWt = qCode.N # This is for a sanity test
    for i in tqdm(range(nTrials)):
        errX,_ = generateError(perr,qCode)
        #print("ERRx")
        synX = HdecX @ errX %2
        bpdX.decode(synX)
        res_X = (errX+bpdX.osdw_decoding)%2
        if (qCode.lz@res_X %2).any():
            error = error + 1
            minWt = min(minWt,np.sum(res_X))
    
    eff_faultRate = error/nTrials
    faultRate = 1 - (1-eff_faultRate)**(1/K)
    faultRate = eff_faultRate
    print("At a Physical error rate of ",perr," ,the net logical error rate is ",(error/nTrials)," effective error rate ",faultRate," minimum bad weight was ",minWt)
    return faultRate,minWt        





#HX,HZ = bivariate_parity_generator_bicycle()
#Qcode = css_code(HX,HZ)
#p = 0.01
#ex,ez = generateError(p,Qcode)
#print("X error is ",ex)
#print("Z error is ",ez)

#err = bposd_decode(Qcode,perr=0.06)


                
if __name__ == "__main__":
    phyError = [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
    logError_noDamage_12 = []
    logError_noDamage_6 = []
    logError_surface = []
    logError_damagedBike = []
    logError_damagedBike_repair = []
    mc_12 = 100000
    mc_6 = 100000
    mc_bike = 100000
    mc_repair = 1000000
    surf = 10000 
    # Gross CODE
    HX,HZ = bivariate_parity_generator_bicycle()
    Qcode_Gross = css_code(HX,HZ)
    #Qcode_Gross.test()
    # Minicycle Code
    ell,m = 6,6
    a1,a2,a3=3,1,2
    b1,b2,b3=3,1,2
    HX,HZ = bivariate_parity_generator_bicycle(ell=ell,m=m,a1=a1,a2=a2,a3=a3,b1=b1,b2=b2,b3=b3)
    Qcode_mini = css_code(HX,HZ)
    # Surface Code (distance 12) --> FOR SANITY TEST
    d = 12
    chain = rep_code(d)
    surface = hgp(h1=chain,h2=chain,compute_distance = True)
    surface.test()
    # Things start to break!
    H1,Z1 = bivariate_parity_generator_bicycle()
    QnewGross = css_code(H1,Z1)
    QBreak = damage_qubit(QnewGross)
    # SeparateStitching rules
    H2,Z2 = bivariate_parity_generator_bicycle()
    QnewGross2 = css_code(H2,Z2)
    QBreak2 = damage_qubit(QnewGross2,symmetry=True)
    for err in phyError:
        eBig,m = bposd_decode(Qcode_Gross,perr=err)
        mc_12 = min(m,mc_12)
        eSmall,m = bposd_decode(Qcode_mini,perr=err)
        mc_6 = min(m,mc_6)
        eSOTA,m = bposd_decode(surface,perr=err)
        surf = min(m,surf)
        eBreak,m = bposd_decode(QBreak,perr=err)
        mc_bike = min(m,mc_bike)
        eRepair,m = bposd_decode(QBreak2,perr=err)
        mc_repair = min(m,mc_repair)
        #eSOTA = 1- (1-eSOTA)**d
        logError_noDamage_12.append(eBig)
        logError_noDamage_6.append(eSmall)
        logError_surface.append(eSOTA)
        logError_damagedBike.append(eBreak)
        logError_damagedBike_repair.append(eRepair)
    print("Monte-carlo obsevred distances in the range of BB-12 = ",mc_12," BB-break = ",mc_bike," BB-6 = ",mc_6," Alternate repair = ",mc_repair)
    # plotting facilities
    plt.plot(phyError,logError_noDamage_12,label="Gross(12)")
    #plt.plot(phyError,logError_noDamage_6,label="[[72,12,6]]")   
    #plt.plot(phyError,logError_surface,label="surface")  
    plt.plot(phyError,logError_damagedBike,label="DamagedGross")  
    plt.plot(phyError,logError_damagedBike_repair,label =" AlternateRepairCode")
    plt.plot(phyError,list(12*np.array(phyError)),label ='PseudoLineThreshold Line_12',linestyle='dashed')
    #plt.plot(phyError,list(6*np.array(phyError)),label ='Threshold Line_6',linestyle='dashed')
    plt.xlabel('Input Physical Error Rate')
    plt.ylabel('Logical/Word error rate')
    plt.ylim(bottom=1e-7)
    plt.legend(fontsize='large')   # Set the font size of the legend
    plt.legend(title='Legend')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='lower right')  
    plt.savefig("FinalResults_OneSided_lowCF.png")
    plt.title("CodeCapacity")
    plt.show()    

