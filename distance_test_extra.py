import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code
from ldpc import mod2
from ldpc.codes import rep_code,ring_code
from bposd.hgp import hgp
from tqdm import tqdm
# computes the minimum Hamming weight of a binary vector x such that 
# stab @ x = 0 mod 2
# logicOp @ x = 1 mod 2
# here stab is a binary matrix and logicOp is a binary vector
def distance_test(stab,logicOp):
	# number of qubits
	n = stab.shape[1]
	# number of stabilizers
	m = stab.shape[0]

	# maximum stabilizer weight
	wstab = np.max([np.sum(stab[i,:]) for i in range(m)])
	# weight of the logical operator
	wlog = np.count_nonzero(logicOp)
	# how many slack variables are needed to express orthogonality constraints modulo two
	num_anc_stab = int(np.ceil(np.log2(wstab)))
	num_anc_logical = int(np.ceil(np.log2(wlog)))
	# total number of variables
	num_var = n + m*num_anc_stab + num_anc_logical

	model = Model()
	model.verbose = 0
	x = [model.add_var(var_type=BINARY) for i in range(num_var)]
	model.objective = minimize(xsum(x[i] for i in range(n)))

	# orthogonality to rows of stab constraints
	for row in range(m):
		weight = [0]*num_var
		supp = np.nonzero(stab[row,:])[0]
		for q in supp:
			weight[q] = 1
		cnt = 1
		for q in range(num_anc_stab):
			weight[n + row*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
		model+= xsum(weight[i] * x[i] for i in range(num_var)) == 0

	# odd overlap with logicOp constraint
	supp = np.nonzero(logicOp)[0]
	weight = [0]*num_var
	for q in supp:
		weight[q] = 1
	cnt = 1
	for q in range(num_anc_logical):
			weight[n + m*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
	model+= xsum(weight[i] * x[i] for i in range(num_var)) == 1
	model.verbose = 0
	model.optimize()

	opt_val = sum([x[i].x for i in range(n)])
	#print("x is ",[x[i].x for i in range(n)])
	outVal = np.array([x[i].x for i in range(n)]).astype(int)
	return int(opt_val),outVal


# Code families to test
# [[144,12,12]]
ell,m = 12,6
a1,a2,a3 = 3,1,2
b1,b2,b3 = 3,1,2
#[[72,12,6]]
#ell,m = 6,6
#a1,a2,a3=3,1,2
#b1,b2,b3=3,1,2

# [[90,8,10]] ---> From decoder_setup.py
#ell,m = 15,3
#a1,a2,a3 = 9,1,2
#b1,b2,b3 = 0,2,7

# [[108,8,10]]
#ell,m = 9,6
#a1,a2,a3 = 3,1,2
#b1,b2,b3 = 3,1,2
# Additional Bigger Codes in Paper(WARNING : Do not run the distance routine with these!)
# [[288,12,18]]
#ell,m = 12,12
#a1,a2,a3 = 3,2,7
#b1,b2,b3 = 3,1,2

# [[360,12,24]]
#ell,m = 30,6
#a1,a2,a3 = 9,1,2
#b1,b2,b3 = 3,25,26

# [[756,16,34]]
#ell,m = 21,18
#a1,a2,a3 = 3,10,17
#b1,b2,b3 = 5,3,19


n = 2*ell*m
n2 = ell*m


# define cyclic shift matrices 
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
hx = np.hstack((A,B))
hz = np.hstack((BT,AT))
print("hx rank is ",mod2.rank(hx))
print("hz rank is ",mod2.rank(hz))
# Killing Qubits (Delete Columns)
Surface = False
if Surface:
	d = 15
	chain = rep_code(d)
	surface = hgp(h1=chain,h2=chain,compute_distance = True)
	#hx = surface.hx
	#hz= surface.hz
	surface.test()
turnOffQubits = [] # Qubits which are defective(This is RANDOM)
HIGH = 1
LOW=0
# TRIAL 1 : COMBINE 3 OPERATORS TO MAKE A SUPEROPERATOR ABC --> FAIL 
#TRAIL 2: AB,BC,CA
#TRIAL 3: First Principles
ALL_THREE = False
TWO_AT_A_TIME = False
SPELL_IT_OUT = True #Read out the defective stabilizers
def damage_qubit(Q,turnOffQubits=[0],symmetry=False,alterning = False,type='data'):
    # Turns off the qubits at specified positions and recalculates the parity matrix
    # TODO: Extend it to multpile qubits
    if type == 'data':
        hxq = Q.hx.copy()
        hzq = Q.hz.copy()
        Q_New = css_code(hxq,hzq)
        Flip = False
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
#hz[turnOffMeasurement,:] = 0
#Check the matrix 
qcode_1=css_code(hx,hz)
patchQubits = [3,60,66,76,77,126]
#patchQubits = []
QdamageZ6_1 = damage_qubit(qcode_1,turnOffQubits=patchQubits,alterning=True)
qcode = damage_qubit(QdamageZ6_1,turnOffQubits=[0,36],type="parity")
print("N is ",qcode.N)
print("K is ",qcode.K)
print("D is ",qcode.D)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx
k = lz.shape[0]

print("Here we print all the logical operators X")
minX = 1000000000
minZ = 1000000000
for rows in lx:
	#print(rows)
	xweight = np.sum(rows)
	#print(" This operator weighs ",xweight)
	readOut = np.where(rows == HIGH)
	#print("This X operator reads ",readOut)
	minX = min(minX,xweight)
print("Now the Z operators")	
for rows in lz:
	#print(rows)
	zweight = np.sum(rows)
	#print(" This operator weighs ",zweight)
	readOut = np.where(rows == HIGH)
	#print("This Z operator reads ",readOut)
	minZ = min(minZ,zweight)
print("The lightest X operator is ",minX," and the lightest Z operator is ",minZ)	
GAUGE = False
##### GAUGE CHECK(IF NOT USING SUPERSTABILIZERS) #######
ALL_3 = False

print(" cHECKING THE X logical operators")
if GAUGE:
	if ALL_3:
		#hxBAD = hxBAD[0]
		#hxBAD = np.sum(hxBAD,axis=0)
		#hzBAD = np.sum(hzBAD,axis=0)
		#hzBAD = hzBAD[0]
		XGaugeCheck = hxBAD @ qcode.lz.T%2
		Xfail = np.unique(np.where(XGaugeCheck == HIGH)[1])
		ZGaugeCheck = hzBAD @ qcode.lx.T%2
		Zfail = np.unique(np.where(ZGaugeCheck == HIGH)[1])
		lxrepair = qcode.lx.copy()
		lzrepair = qcode.lz.copy()
		lxrepair = np.delete(lxrepair,Xfail,axis=0)
		lzrepair = np.delete(lzrepair,Zfail,axis=0)
		print("After deleting the rank becomes ",mod2.rank(lxrepair@lzrepair.T %2))
		DEBUG = True
		if DEBUG:
			FIXER_UPPER =Zfail[-1]
			print("Common link is ",FIXER_UPPER)
			Xfail = Xfail[:-1]
			Zfail = Zfail[:-1]
			lX = qcode.lx.copy()
			lZ= qcode.lz.copy()
			for badplaces in Xfail:
				print("Badplace being fixed ",badplaces)
				lZ[badplaces] = (lZ[badplaces]+lZ[FIXER_UPPER]) %2
			#lX = np.delete(lX,FIXER_UPPER,axis = 0)
			for badplaces in Zfail:
				print("Badplace being fixed Z",badplaces)
				lX[badplaces] =(lX[badplaces]+lX[FIXER_UPPER])%2

			#lZ = np.delete(lZ,FIXER_UPPER,axis = 0)
			# Try again
			XGaugeCheck = hxBAD @ lZ.T%2
			Xfail = np.where(XGaugeCheck == HIGH)[1]
			ZGaugeCheck = hzBAD @ lX.T%2
			Zfail = np.where(ZGaugeCheck == HIGH)[1]
			lxrepair = lX.copy()
			lzrepair = lZ.copy()
			lxrepair = np.delete(lxrepair,Xfail,axis=0)
			lzrepair = np.delete(lzrepair,Zfail,axis=0)
			print("After deleting the rank becomes ",mod2.rank(lxrepair@lzrepair.T %2))
			print("Hx check")
			# GAUGE FIXED
			qcode.K = mod2.rank(lxrepair@lzrepair.T %2)
			qcode.lx= lxrepair
			qcode.lz= lzrepair
			qcode.test()

	else:
		hxBAD[0] = hxBAD[0] + hxBAD[2]
		hxBAD[1] = hxBAD[1] + hxBAD[2]
		hxBAD = np.delete(hxBAD,2,axis=0)	
		hzBAD[0] = hzBAD[0] + hzBAD[2]
		hzBAD[1] = hzBAD[1] + hzBAD[2]
		hzBAD = np.delete(hzBAD,2,axis=0)
		XGaugeCheck = hxBAD @ qcode.lz.T%2
		Xfail = np.where(XGaugeCheck == HIGH)[1]
		ZGaugeCheck = hzBAD @ qcode.lx.T%2
		Zfail = np.where(ZGaugeCheck == HIGH)[1]
	#print("Hxbas shappe is ",hxBAD.shape)
	#XGaugeCheck = hxBAD @ qcode.lz.T%2
	print("XGaugeCheck returns ",XGaugeCheck)
	#Read out the columns where the value is high
	#Xfail = np.where(XGaugeCheck == HIGH)[1]
	#Xfail = np.where(XGaugeCheck == HIGH)
	print("Gauge Check X fails at ",np.unique(Xfail))
	print(" Checking the Z logical operators ")
	#hzBAD = np.sum(hzBAD,axis=0)
	#ZGaugeCheck = hzBAD @ qcode.lx.T%2
	print("ZGaugeCheck returns ",ZGaugeCheck)
	#Read out the columns where the value is high
	#Zfail = np.where(ZGaugeCheck == HIGH)[1]
	#Zfail = np.where(ZGaugeCheck == HIGH)
	print("Gauge Check fails at ",np.unique(Zfail))
	totalGaugeSpace = set(np.unique(Xfail)) |  set(np.unique(Zfail))
	print("This is going to go ",totalGaugeSpace)
	# Fail Correction
	FAIL_SAFE = False
	if FAIL_SAFE:
		lzNew = qcode.lz.copy()
		lxNew = qcode.lx.copy()
		nonCommuters = list(totalGaugeSpace)
		#Delete the gaugeable things
		lzNew = np.delete(lzNew,nonCommuters,axis=0)
		lxNew = np.delete(lxNew,nonCommuters,axis=0)
		newK = qcode.K - len(nonCommuters)
		# Create the superopertaor 
		#hxSuper = np.sum(hxBAD,axis=0)%2
		hxSuper = hxBAD
		print("Super shape is ",hxBAD.shape)
		#hzSuper = np.sum(hzBAD,axis=0)%2
		hzSuper = hzBAD
		#qcode.hx = np.vstack((hx,hxSuper))
		#HZsUPER = np.sum(hzBAD,axis=0)%2
		#qcode.hz = np.vstack((hz,hzSuper))
		qcode.lx = lxNew
		qcode.lz = lzNew
		qcode.test()
#Turn off the logical orders
#turnOffDim = 
print("Print the Z logical check operators")
print("Logical Check")
## X_L^2 = Z_L^2 = I
x_chk = qcode.lx @ qcode.lz.T %2
z_chk = qcode.lz @ qcode.lx.T %2
print("Logical shape is ",qcode.lx.shape)
#print("X identity is ",(x_chk))
#print("Z identity is ",(z_chk))
##### GAUGE CHECK(IF NOT USING SUPERSTABILIZERS) #######
print('Computing code distance...')
# We compute the distance only for Z-type logical operators (the distance for X-type logical operators is the same)
# by solving an integer linear program (ILP). The ILP looks for a minimum weight Pauli Z-type operator which has an even overlap with each X-check 
# and an odd overlap with logical-X operator on the i-th logical qubit. Let w_i be the optimal value of this ILP. 
# Then the code distance for Z-type logical operators is dZ = min(w_1,…,w_k).
d = n
qubitix = []
qubitiz = []

DISABLE = False
if Surface:
	qcode = surface
	hz1 = qcode.hz
	hx1 = qcode.hx
	# Kill 1 qubit
	hz1[0,:] = 0 * hz1[0,:]
	qcode = css_code(hx1,hz1)
	qcode.test()
	# Kill 1 parity qubit here
if not DISABLE:
	k = qcode.K
	N = qcode.N
	dim = (k,N)
	hxSmall = np.zeros(dim,dtype=int)
	hzSmall = np.zeros(dim,dtype=int)
	hx = qcode.hx.copy()
	hz = qcode.hz.copy()
	lx = qcode.lx.copy()
	lz = qcode.lz.copy()
	for i in range(k):
		w2,hzSmalli = distance_test(hz,lz[i,:])
		w1,hxSmalli = distance_test(hx,lx[i,:])
		
		print('Logical qubitX=',i,'Distance=',w1)
		print('Logical qubitZ=',i,'Distance=',w2)
		qubitix.append(w1)
		qubitiz.append(w2)
		hxSmall[i] = hxSmalli
		hzSmall[i] =  hzSmalli
		w1 = 10000
		d = min(d,w1,w2)

print('Code parameters: n,k,d=',n,k,d)
#print("hx is ",hx)
print("All X distances are as follows ",qubitix)
print("All Z distances are as follows ",qubitiz)
qcode.test()
print("k is ",lz.shape)
#Remove 1 from the weight because one qubit is gone
if GAUGE:
	broken_rows_x_DEBUG[:,1] = LOW
	broken_rows_z_DEBUG[:,1] = LOW
	# 3 PRODUCT WEIGHT
	broken_rows_x_DEBUG = np.sum(broken_rows_x_DEBUG,axis=0)
	broken_rows_x_DEBUG = broken_rows_x_DEBUG%2
	broken_rows_z_DEBUG = np.sum(broken_rows_z_DEBUG,axis=0)
	broken_rows_z_DEBUG = broken_rows_z_DEBUG%2
	print("X weights ",np.sum(broken_rows_x_DEBUG))
	print("Z weights ",np.sum(broken_rows_z_DEBUG))
	print(" Sanity Checks")
#print(mod2.rank(hxSmall@hzSmall.T%2))
EXPERIMENT = False
if EXPERIMENT:
	print("Simulating killing parity qubits on  Code")
	Surface = False
	if Surface:
		d = 9
		chain = ring_code(d)
		#chain = rep_code(d)
		qcode = hgp(h1=chain,h2=chain,compute_distance = True)
		#hx = surface.hx
		#hz= surface.hz
		#surface.test()
	kmax = qcode.K +1 
	nTrials = 10000
	nKill = 3
	# Choose nKill parity to kill .The first WLOG can be 0 . For a first Order check sample randomly
	# Small Code
	nChoose = int(qcode.N/2) 
	#signature = []
	flip = 0
	bad_places = []
	for i in tqdm(range(nTrials)):
		signature =[]
		turnOfflines = np.random.choice(np.arange(1,nChoose),size=nKill-1,replace= False)
		turnOfflines = np.append(turnOfflines,0)
		hx = qcode.hx.copy()
		hz = qcode.hz.copy()
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
			bad_places.append(turnOfflines[0])
			kmax = k_sample
	print("Maximum qubits is ",kmax)	
	print("Total number of flips in 10000 : ",flip)		
	print("code test")
	qcode.test()
	bad_places = np.unique(bad_places)
	print(" Bad turn off locations are ",len(bad_places))

