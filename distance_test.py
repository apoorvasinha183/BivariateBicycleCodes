import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code

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
	return int(opt_val)


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
# Killing Qubits (Delete Columns)
turnOffQubits = [0] # Qubits which are defective(This is RANDOM)
HIGH = 1
LOW=0
# TRIAL 1 : COMBINE 3 OPERATORS TO MAKE A SUPEROPERATOR ABC --> FAIL 
#TRAIL 2: AB,BC,CA
#TRIAL 3: First Principles
ALL_THREE = False
TWO_AT_A_TIME = True
SPELL_IT_OUT = False #Read out the defective stabilizers
for defects in turnOffQubits:
	# Find all rows where the turnOffQubits are high.To figure out the connectivity
	# X-Z syndromes
	
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
		#hx[rows_to_be_deleted_x[0]] = (hx_old[rows_to_be_deleted_x[1]]+hx_old[rows_to_be_deleted_x[2]])%2
		hz[rows_to_be_deleted_z[0]] = (hz_old[rows_to_be_deleted_z[1]]+hz_old[rows_to_be_deleted_z[2]])%2
		# BC
		#hx[rows_to_be_deleted_x[1]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[2]])%2
		hz[rows_to_be_deleted_z[1]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[2]])%2
		# CA
		#hx[rows_to_be_deleted_x[2]] = (hx_old[rows_to_be_deleted_x[0]]+hx_old[rows_to_be_deleted_x[1]])%2
		#hz[rows_to_be_deleted_z[2]] = (hz_old[rows_to_be_deleted_z[0]]+hz_old[rows_to_be_deleted_z[1]])%2
		hz[rows_to_be_deleted_z[2]] = 0* hz[rows_to_be_deleted_z[2]]
		# Delete the extra row
		#hx = np.delete(hx,rows_to_be_deleted_x[2],axis=0)
		#hz = np.delete(hz,rows_to_be_deleted_z[2],axis=0)
	replace_x = rows_to_be_deleted_x[0]
	replace_z = rows_to_be_deleted_z[0]
	print("To be deleted ",rows_to_be_deleted_x)
	#hx = np.delete(hx,rows_to_be_deleted_x[1:],axis = 0 )
	#hz = np.delete(hz,rows_to_be_deleted_z[1:],axis = 0 )
	#hx[:,defects] =LOW
	#hz[:,defects] = LOW
	#hx[replace_x] = np.sum(broken_rows_x,axis=0) %2
	#hz[replace_z] = np.sum(broken_rows_z,axis=0) %2
	# Now kill the qubit
	hx = np.delete(hx,defects,axis=1)
	hz = np.delete(hz,defects,axis=1)
	#hx[:,defects] = LOW
	#hz[:,defects] = LOW

	#hx = hx[:,1:] 
	#hz = hz[:,1:]
# Find out where the code has gone bad
print("Matrix shape is ",hx.shape)
mat_check = ((hz@hx.T %2)+(hx@hz.T %2))%2
length,breadth = mat_check.shape
for i in range(length):
	print(mat_check[i])
print("Bad matrix above")
#	hz[:,defects] = 0
# Killing Measurements (Delete Rows FROM one row !)
turnOffMeasurement = 0
#hx[turnOffMeasurement,:] = 0
#hz[turnOffMeasurement,:] = 0
#Check the matrix 
qcode=css_code(hx,hz)
print("N is ",qcode.N)
print("K is ",qcode.K)
print("D is ",qcode.D)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx
k = lz.shape[0]


print('Computing code distance...')
# We compute the distance only for Z-type logical operators (the distance for X-type logical operators is the same)
# by solving an integer linear program (ILP). The ILP looks for a minimum weight Pauli Z-type operator which has an even overlap with each X-check 
# and an odd overlap with logical-X operator on the i-th logical qubit. Let w_i be the optimal value of this ILP. 
# Then the code distance for Z-type logical operators is dZ = min(w_1,…,w_k).
d = n
qubitix = []
qubitiz = []
#TODO: When css is destroyed does hx,hz separation make no mathematical sense? Use the collated l and H operator in that case
#TODO: Turning off checks makes no difference??
DISABLE =  False
if not DISABLE:
	for i in range(k):
		w1 = distance_test(hx,lx[i,:])
		w2 = distance_test(hz,lz[i,:])
		print('Logical qubitX=',i,'Distance=',w1)
		print('Logical qubitZ=',i,'Distance=',w1)
		qubitix.append(w1)
		qubitiz.append(w2)
		d = min(d,w1,w2)

print('Code parameters: n,k,d=',n,k,d)
print("hx is ",hx)
print("All X distances are as follows ",qubitix)
print("All Z distances are as follows ",qubitiz)
qcode.test()
print("k is ",lz.shape)
#Remove 1 from the weight because one qubit is gone
broken_rows_x_DEBUG[:,1] = LOW
broken_rows_z_DEBUG[:,1] = LOW
# 3 PRODUCT WEIGHT
broken_rows_x_DEBUG = np.sum(broken_rows_x_DEBUG,axis=0)
broken_rows_x_DEBUG = broken_rows_x_DEBUG%2
broken_rows_z_DEBUG = np.sum(broken_rows_z_DEBUG,axis=0)
broken_rows_z_DEBUG = broken_rows_z_DEBUG%2
print("X weights ",np.sum(broken_rows_x_DEBUG))
print("Z weights ",np.sum(broken_rows_z_DEBUG))