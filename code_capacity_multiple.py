# Code Capacity Simulations
# Simple Random X and Z errors of the Physical Qubit length are performed
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

def damage_qubit(Q,turnOffQubits=[0],symmetry=False,alterning = False,type='data'):
    # Turns off the qubits at specified positions and recalculates the parity matrix
    # TODO: Extend it to multpile qubits
    if type == 'data':
        hxq = Q.hx.copy()
        hzq = Q.hz.copy()
        Q_New = css_code(hxq,hzq)
        Flip = True
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
                #hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
                #hz[rows_to_be_deleted_z[2]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[1]])%2
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
            #hxBAD = hx[rows_to_be_deleted_x]
            #hzBAD = hz[rows_to_be_deleted_z]
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
            #hx = np.delete(hx,defects,axis=1)
            #hz = np.delete(hz,defects,axis=1)
            #hxBAD = np.delete(hxBAD,defects,axis=1)
            #hzBAD = np.delete(hzBAD,defects,axis=1)
            #hx[:,defects] = LOW
            #hz[:,defects] = LOW

            #hx = hx[:,1:] 
            #hz = hz[:,1:]
            if alterning:
                Flip = not Flip
        # Deleted at the end because the numbering goes out of sync
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

def bposd_decode(qCode,type='z',perr=0.01,nTrials = 100):
    #Runs Roffee's BP-OSD Decoder to infer the syndrome
    #print("The type I see is ",type)
    if type == 'z':
    #    print("Z type")
        HdecX = qCode.hz
        logicalX = qCode.lz
    else:
    #    print("X type")
        HdecX = qCode.hx
        logicalX = qCode.lx    
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
    for i in range(nTrials):
        errX,_ = generateError(perr,qCode)
        #print("ERRx")
        synX = HdecX @ errX %2
        bpdX.decode(synX)
        res_X = (errX+bpdX.osdw_decoding)%2
        if (logicalX@res_X %2).any():
            error = error + 1
            minWt = min(minWt,np.sum(res_X))
    
    eff_faultRate = error/nTrials
    faultRate = 1 - (1-eff_faultRate)**(1/K)
    faultRate = eff_faultRate
    #print("At a Physical error rate of ",perr," ,the net logical error rate is ",(error/nTrials)," effective error rate ",faultRate," minimum bad weight was ",minWt)
    return faultRate,minWt        





#HX,HZ = bivariate_parity_generator_bicycle()
#Qcode = css_code(HX,HZ)
#p = 0.01
#ex,ez = generateError(p,Qcode)
#print("X error is ",ex)
#print("Z error is ",ez)

#err = bposd_decode(Qcode,perr=0.06)
#def datagen(nTrials=10000,perr=0.01,lerr=0.01,type='z',)

                
if __name__ == "__main__":
    args = sys.argv[1:]
    
    if args[0] == "-type":
        type = args[1]
    else:
        type= 'z'
    if args[2] == "-dmg":
        snd = args[3]
    else:
        snd = 5        
    print("Simulation of type errors "+ type)
    print("Error place ",snd)
    nTrails = 1000
    damageQubits = [0]
    damageQubits.append(int(snd))
    phyError = [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11]
    #phyError =[0.09]
    logError_noDamage_12 = []
    logError_noDamage_6 = []
    logError_surface = []
    logError_damagedBike = []
    logError_damagedBike_repair = []
    logError_XZ_repair = []
    logError_paritySingleLoss = [] # Delete a single check qubit
    logError_parityDoubleLoss =[] # Delete double qubits
    logError_parityDoubleLossBad = [] # Delete the sensitive pair
    logError_parityDoubleLossBad_fix1 = [] # Delete the sensitive pair -FIX 1
    logError_parityDoubleLossBad_fix2 = [] # Delete the sensitive pair -FIX 2
    mc_12 = 100000
    mc_6 = 100000
    mc_bike = 100000
    mc_repair = 1000000
    mc_alternate = 100000000
    surf = 10000 
    mc_1parity = 10000000
    mc_2parity = 1000000
    mc_2paritybad = 1000000
    mc_2parityfix1 = 100000000
    mc_2parityfix2 = 100000000
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
    #d = 12
    #chain = rep_code(d)
    #surface = hgp(h1=chain,h2=chain,compute_distance = True)
    #surface.test()
    # Things start to break!
    H1,Z1 = bivariate_parity_generator_bicycle()
    QnewGross = css_code(H1,Z1)
    print("Stitched asymmetrically")
    QBreak = damage_qubit(QnewGross,turnOffQubits=damageQubits,alterning=False)
    # Multiple stitches alternate between X/Z
    H4,Z4 = bivariate_parity_generator_bicycle()
    QnewGross3 = css_code(H4,Z4)
    print("Stitched asymmetrically but in X/Z fashion")
    QBreak3 = damage_qubit(QnewGross3,turnOffQubits=damageQubits,alterning=True)
    # SeparateStitching rules
    H2,Z2 = bivariate_parity_generator_bicycle()
    QnewGross2 = css_code(H2,Z2)
    print("Stitched symmetrically")
    QBreak2 = damage_qubit(QnewGross2,turnOffQubits=damageQubits,symmetry=True)
    # Single Parity Loss
    H5,Z5 = bivariate_parity_generator_bicycle()
    Q5 = css_code(H5,Z5)
    QDef1Parity = damage_qubit(Q5,turnOffQubits=[0],type="parity")
    print("1 parity defect added")
    # Double Parity Loss(non intrusive)
    H6,Z6 = bivariate_parity_generator_bicycle()
    Q6 = css_code(H6,Z6)
    QDef2Parity = damage_qubit(Q6,turnOffQubits=[0,11],type="parity")
    print("2 parity defects added(Non Intrusive)")
    # Double Parity Loss( intrusive)
    H7,Z7 = bivariate_parity_generator_bicycle()
    QDef2BadParity = css_code(H7,Z7)
    QDef2ParityBad = damage_qubit(Q6,turnOffQubits=[0,36],type="parity") # Searcged
    print("2 parity defects added( Intrusive)")
    # Idea to fix 
    #Idea 1 All Same Parity - I think all distances will dive by 6 for one parity
    H8,Z8 = bivariate_parity_generator_bicycle()
    QDef2BadParity_F = css_code(H8,Z8)
    # [0,36] is the only bad measurement combination in [[144,12,12]]
    # Parity Qubit 0 is connected to 3,60,66,76,77,126
    patchQubits = [3,60,66,76,77,126]
    QdamageZ6_1 = damage_qubit(QDef2BadParity_F,turnOffQubits=patchQubits,alterning=False)
    
    # Disable Parities
    QdamageZZ = damage_qubit(QdamageZ6_1,turnOffQubits=[0,36],type="parity")
    print(" All Qubits of same parity used for fix(may fail)")
    print("Testing this code-Fix1 ")
    QdamageZZ.test()
    # Idea 2 -Alternating Fix - Take a 3-3 penalty in both distances
    H9,Z9 = bivariate_parity_generator_bicycle()
    QDef2BadParity_F1 = css_code(H9,Z9)
    QdamageZ6_2 = damage_qubit(QDef2BadParity_F1,turnOffQubits=patchQubits,alterning=True)
    # Disable Parities
    QdamageZX = damage_qubit(QdamageZ6_2,turnOffQubits=[0,36],type="parity")
    print("  Qubits of alternating parity used for fix(may fail)")
    print("Testing this code-Fix2 ")
    QdamageZX.test()
    DEBUG = False
    for err in tqdm(phyError):
        if not DEBUG:
            print("[[144,12,12]]")
            eBig,m = bposd_decode(Qcode_Gross,perr=err,type=type,nTrials=nTrails)
            mc_12 = min(m,mc_12)
            #print("[[72,12,6]]")
            #eSmall,m = bposd_decode(Qcode_mini,perr=err,nTrials=nTrails)
            #mc_6 = min(m,mc_6)
            #eSOTA,m = bposd_decode(surface,perr=err,nTrials=nTrails)
            #surf = min(m,surf)
            print("Asymmetric Repair ZZ")
            eBreak,m = bposd_decode(QBreak,perr=err,type=type,nTrials=nTrails)
            mc_bike = min(m,mc_bike)
            print("Asymmetric Repair ZX")
            eBreak3,m = bposd_decode(QBreak3,perr=err,type=type,nTrials=nTrails)
            mc_alternate = min(m,mc_alternate)
            print("Symmetric Repair")
            eRepair,m = bposd_decode(QBreak2,perr=err,type=type,nTrials=nTrails)
            mc_repair = min(m,mc_repair)
            logError_XZ_repair.append(eBreak3)
            logError_damagedBike.append(eBreak)
            logError_noDamage_12.append(eBig)
            logError_damagedBike_repair.append(eRepair)
        #eSOTA = 1- (1-eSOTA)**d
            print("1 parity loss")
            eParity1,m = bposd_decode(QDef1Parity,perr=err,type=type,nTrials=nTrails)
            logError_paritySingleLoss.append(eParity1)
            mc_1parity = min(m,mc_1parity)
            print("2 parity losses but not bad")
            eParity2,m = bposd_decode(QDef2Parity,perr=err,type=type,nTrials=nTrails)
            logError_parityDoubleLoss.append(eParity2)
            mc_2parity = min(m,mc_2parity)
            print("2 parity losses but sensitive")
            eParity3,m = bposd_decode(QDef2ParityBad,perr=err,type=type,nTrials=nTrails)
            logError_parityDoubleLossBad.append(eParity3)
            mc_2paritybad = min(m,mc_2paritybad)
        print("2 parity bad fix 1")
        eParity4,m = bposd_decode(QdamageZZ,perr=err,type=type,nTrials=nTrails)
        logError_parityDoubleLossBad_fix1.append(eParity4)
        mc_2parityfix1 = min(m,mc_2parityfix1)
        print("2 parity bad fix 2")
        eParity5,m = bposd_decode(QdamageZX,perr=err,type=type,nTrials=nTrails)
        logError_parityDoubleLossBad_fix2.append(eParity5)
        mc_2parityfix2 = min(m,mc_2parityfix2)
        #logError_noDamage_6.append(eSmall)
        #print("Surface")
        #logError_surface.append(eSOTA)
        
    print("Monte-carlo obsevred distances in the range of BB-12 = ",mc_12," BB-break = ",mc_bike," XZ repair = ",mc_alternate," Alternate repair = ",mc_repair," One Bad Parity = ",mc_1parity," Two NoniNTRUSIVE Pairites =  ",mc_2parity," Two bad parities bad = ",mc_2paritybad," Bad parity Fix 1 = ",mc_2parityfix1," Bad Parity Fix 2 = ",mc_2parityfix2)
    # plotting facilities
    if not DEBUG:
        plt.plot(phyError,logError_noDamage_12,label="Gross(12)")
        #plt.plot(phyError,logError_noDamage_6,label="[[72,12,6]]")   
        #plt.plot(phyError,logError_surface,label="surface")  
        #plt.plot(phyError,logError_damagedBike,label="DamagedGrossZZ")  
        #plt.plot(phyError,logError_XZ_repair,label="DamagedGrossXZ")
        #plt.plot(phyError,logError_damagedBike_repair,label =" AlternateRepairCode")
        plt.plot(phyError,logError_paritySingleLoss,label = "Single Bad Parity Qubit")
        plt.plot(phyError,logError_parityDoubleLoss,label = "Two Bad Parity Qubits(OK)")
        plt.plot(phyError,logError_parityDoubleLossBad,label = "Two Bad Parity Qubits")
        plt.plot(phyError,logError_parityDoubleLossBad_fix1,label = "Two Bad Parity Qubits-Fix 1")
        plt.plot(phyError,logError_parityDoubleLossBad_fix2,label = "Two Bad Parity Qubits-Fix 2")
        #plt.plot(phyError,list(np.clip(12*np.array(phyError),None,1)),label ='PseudoLineThreshold Line_12',linestyle='dashed')
        #plt.plot(phyError,1-(1-np.array(phyError))**12,label ='PseudoLineThreshold Line_12',linestyle='dashed') # Better definition of threshold
        #plt.plot(phyError,list(6*np.array(phyError)),label ='Threshold Line_6',linestyle='dashed')
        plt.xlabel('Input Physical Error Rate')
        plt.ylabel('Logical error rate') # If you use block error rate ,change the threshold line
        #plt.ylim(bottom=1e-7)
        plt.legend(fontsize='large')   # Set the font size of the legend
        plt.legend(title='Legend')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left')  
        if type == 'x':
            fn = "FinalResults_X_Error_parityexperiment_"
            for defects in damageQubits:
                fn += str(defects)+"|"
            fn += ".png"    
        else:
            fn = "FinalResults_Z_Error_parityexperiment_"
            for defects in damageQubits:
                fn += str(defects)+"|"
            fn += ".png"   
        plt.savefig(fn)
        plt.title("CodeCapacity")
    #plt.show()    

