import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code
from ldpc import mod2
from ldpc.codes import rep_code,ring_code
from bposd.hgp import hgp
from tqdm import tqdm
from defect_parity_generator import *
import logging
from tqdm import tqdm
from bposd.css_decode_sim import css_decode_sim
from ldpc.codes import rep_code
from ldpc.codes import ring_code
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
	#model.objective = xsum(x[i] for i in range(n))
	#print(model.verbose)
	#sys.exit()
	#if 1:
	#	assa

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
	model.verbose = False
	#model.log_level = 1 if model.verbose else 0
	model.solver.set_verbose(0)
	model.optimize()

	opt_val = sum([x[i].x for i in range(n)])
	#print("x is ",[x[i].x for i in range(n)])
	outVal = np.array([x[i].x for i in range(n)]).astype(int)
	return int(opt_val),outVal

def Calculate_Distance(qcode):
	qubitix = []
	qubitiz = []
	k = qcode.K
	N = qcode.N
	dim = (k,N)
	hxSmall = np.zeros(dim,dtype=int)
	hzSmall = np.zeros(dim,dtype=int)
	hx = qcode.hx.copy()
	hz = qcode.hz.copy()
	lx = qcode.lx.copy()
	lz = qcode.lz.copy()
	d = 10000
	target_runs = k
	# Shuffle the rows of lx and lz
	#shuffled_indices = np.random.permutation(k)
	lx_shuffled = lx.copy()
	lz_shuffled = lz.copy()
	#k =1
	pbar = tqdm(range(k),ncols=0)
	#Randomize the order of the lx and lz matrices
	for i in pbar:
		dx = 1000
		dz = 1000
		w1,hxSmalli = distance_test(hx,lx_shuffled[i,:])
		#w1,hxSmalli = (2,3)
		w2,hzSmalli = distance_test(hz,lz_shuffled[i,:])
		#print('Logical qubitX=',i,'Distance=',w1)
		#print('Logical qubitZ=',i,'Distance=',w2)
		qubitix.append(w1)
		qubitiz.append(w2)
		hxSmall[i] = hxSmalli
		hzSmall[i] =  hzSmalli
		#w1 = 10000
		dx = min(dx,w1)
		dz = min(dz,w2)
		d = min(d,w1,w2)
		pbar.set_description(f"d_min: {d};dx:{dz};dz:{dx}")
	pbar.close()
	print("Z distances ",qubitix)
	print("X distances ",qubitiz)
	print("Minimum distance is ",d)
	#Print the erroneous syndromes these are logical operators
	for i in range(k):
		#Print the z- syndrome
		print(f'Ztype error{i} is {np.where(hxSmall[i]==1)[0]}')
	for i in range(k):
		# Print the x-syndrome
		print(f'Xtype error{i} is {np.where(hzSmall[i]==1)[0]}')

	

def Sample_Distance(qCode,type='z',perr=0.01,nTrials = 100,disable= False):
	#Runs Roffee's BP-OSD Decoder to infer the syndrome
	#print("The type I see is ",type)
	if type == 'z':
	#    print("Z type")
		HdecX = qCode.hz.copy()
		logicalX = qCode.lz.copy()
	else:
	#    print("X type")
		HdecX = qCode.hx.copy()
		logicalX = qCode.lx.copy()    
	channel_probsX = perr
	#channel_probsX = 0.09
	K = qCode.K
	print('The number of logical qubits is ',K)
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
	minWt =50
	pbar = tqdm(range(nTrials),ncols=0,disable=disable)
	for i in pbar:
		errX,_ = generateError(perr,qCode)
		#print("ERRx")
		synX = HdecX @ errX %2
		bpdX.decode(synX)
		res_X = (errX+bpdX.osdw_decoding)%2
		if (logicalX@res_X %2).any():
			error = error + 1
			minWt = min(minWt,np.sum(res_X))
		pbar.set_description(f"d_min: {minWt}; errors: {error}; trials: {i};type: {type};perr: {perr}")
	
	eff_faultRate = error/nTrials
	faultRate = 1 - (1-eff_faultRate)**(1/K)
	#faultRate = eff_faultRate
	
	return faultRate,minWt,error        
	
		


if __name__ == "__main__":
	codeName = code_dict(name="d10-90")
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_One = bikeCode
	DEFECT = [0,1,2,3]
	damagedBikeCode_One = damage_qubit(bikeCode,turnOffQubits=DEFECT,symmetry=False,flip=True,test=True)
	print("hx rank",mod2.rank(damagedBikeCode_One.hx))
	print("hz rank",mod2.rank(damagedBikeCode_One.hz))
	# Find the logical operators of the code
	lx1 = damagedBikeCode_One.lx.copy()
	lz1 = damagedBikeCode_One.lz.copy()
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_Two = damage_qubit(bikeCode,turnOffQubits=DEFECT,symmetry=False,test=True)
	# Find the logical operators of the code
	lx2 = damagedBikeCode_Two.lx.copy()
	lz2 = damagedBikeCode_Two.lz.copy()
	#experiment with number of qubits
	print(f'Number of qubits combination 1 :{mod2.rank(lx1@lz2.T%2)};Number of qubits in combination 2:{mod2.rank(lx2@lz1.T%2)}')
	bikeCode_2 = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_Three = damage_qubit(bikeCode_2,turnOffQubits=DEFECT,symmetry=True)
	damagedBikeCode_Three.test()
	print("Passing logicals")
	damagedBikeCode_Three.lx = lx1
	damagedBikeCode_Three.lz = lz2
	damagedBikeCode_Three.K = damagedBikeCode_One.K
	damagedBikeCode_Three.test()
	print("hx rank",mod2.rank(damagedBikeCode_Three.hx))
	print("hz rank",mod2.rank(damagedBikeCode_Three.hz))
	distinctz = []
	distinctx = []
	for rows in damagedBikeCode_Three.hx:
		if np.sum(rows) not in distinctx:
			distinctx.append(np.sum(rows))
	for rows in damagedBikeCode_Three.hz:
		if np.sum(rows) not in distinctz:
			distinctz.append(np.sum(rows))
	print(f"distinctxweights :{distinctx}")
	print(f"distinctzweights :{distinctz}")
	#bikeCode = bivariate_parity_generator_bicycle(codeName)
	#damagedBikeCode_Two = damage_qubit(bikeCode,turnOffQubits=[0,1,22])
	Calculate_Distance(damagedBikeCode_Three)
	bikeCode_3 = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_Four = damage_qubit(bikeCode_3,turnOffQubits=DEFECT,alterning=True)
	damagedBikeCode_Four.test()
	Calculate_Distance(damagedBikeCode_Four)
	# Try Roffee's Routine Check
	osd_options={
		'error_rate': 0.05,
		'target_runs': 1000000,
		'xyz_error_bias': [1, 0, 0],
		'output_file': 'test.json',
		'bp_method': "ms",
		'ms_scaling_factor': 0,
		'osd_method': "osd_cs",
		'osd_order': 42,
		'channel_update': None,
		'seed': 42,
		'max_iter': 0,
		'output_file': "test.json"
		}
	surf = False
	if surf:
		h=ring_code(9)
		surface_code=hgp(h1=h,h2=h,compute_distance=True) #nb. set compute_distance=False for larger codes
		surface_code.test()
		# Cause defect randomly
		print(surface_code.N)
		defects = np.random.randint(0,surface_code.N)
		defects = 0
		#print(random_integer)
		# Find the defecgtive qubits in X/Z and disable them
		rows_to_be_deleted_x = np.where(surface_code.hx[:,defects]==1)[0]
		rows_to_be_deleted_z = np.where(surface_code.hz[:,defects] == 1)[0]
		print(f'rows to deleted x is {rows_to_be_deleted_x}')
		print(f'rows to deleted z is {rows_to_be_deleted_z}')
		#Only Z (lz)
		saved_x = []
		saved_z = []
		for i in range(2):
			if i%2 :
				#surface_code.hx[rows_to_be_deleted_x[i]] = (surface_code.hx[rows_to_be_deleted_x[i]] + saved_x) %2
				#surface_code.hx[rows_to_be_deleted_x[i]] = 0
				#surface_code.hz[rows_to_be_deleted_z[i]] = (surface_code.hz[rows_to_be_deleted_z[i]] + saved_z) %2
				surface_code.hz[rows_to_be_deleted_z[i]] = 0
			else:
				saved_x = surface_code.hx[rows_to_be_deleted_x[i]].copy()
				#surface_code.hx[rows_to_be_deleted_x[i]] = 0
				saved_z = surface_code.hz[rows_to_be_deleted_z[i]].copy()
				surface_code.hz[rows_to_be_deleted_z[i]] = 0
		#Delte the qubit
		surface_code.hx = np.delete(surface_code.hx,defects,axis=1)	
		surface_code.hz = np.delete(surface_code.hz,defects,axis=1)			   
		surface_code=css_code(surface_code.hx,surface_code.hz)

		#surface_code.hx[surface_code.hx[:,random_integer]== 1] = 0
		#surface_code.hz[surface_code.hz[:,random_integer]== 1] = 0
		print("This broken surface code has these properties")
		if surface_code.test():
			Calculate_Distance(surface_code)
		lz_0 = surface_code.lz.copy()
		#Only X(lx)
		surface_code=hgp(h1=h,h2=h,compute_distance=True) #nb. set compute_distance=False for larger codes
		surface_code.test()
		saved_x = []
		saved_z = []
		for i in range(2):
			if i%2 :
				#surface_code.hx[rows_to_be_deleted_x[i]] = (surface_code.hx[rows_to_be_deleted_x[i]] + saved_x) %2
				surface_code.hx[rows_to_be_deleted_x[i]] = 0
				#surface_code.hz[rows_to_be_deleted_z[i]] = (surface_code.hz[rows_to_be_deleted_z[i]] + saved_z) %2
				#surface_code.hz[rows_to_be_deleted_z[i]] = 0
			else:
				saved_x = surface_code.hx[rows_to_be_deleted_x[i]].copy()
				surface_code.hx[rows_to_be_deleted_x[i]] = 0
				saved_z = surface_code.hz[rows_to_be_deleted_z[i]].copy()
				#surface_code.hz[rows_to_be_deleted_z[i]] = 0
		#Delte the qubit
		surface_code.hx = np.delete(surface_code.hx,defects,axis=1)	
		surface_code.hz = np.delete(surface_code.hz,defects,axis=1)			   
		surface_code=css_code(surface_code.hx,surface_code.hz)

		#surface_code.hx[surface_code.hx[:,random_integer]== 1] = 0
		#surface_code.hz[surface_code.hz[:,random_integer]== 1] = 0
		print("This broken surface code has these properties")
		if surface_code.test():
			Calculate_Distance(surface_code)
		lx_0 = surface_code.lx.copy()
		#Extract logicals
		#Both
		#Exchange logicals
		surface_code=hgp(h1=h,h2=h,compute_distance=True) #nb. set compute_distance=False for larger codes
		surface_code.test()
		saved_x = []
		saved_z = []
		for i in range(2):
			if i%2 :
				#surface_code.hx[rows_to_be_deleted_x[i]] = (surface_code.hx[rows_to_be_deleted_x[i]] + saved_x) %2
				surface_code.hx[rows_to_be_deleted_x[i]] = 0
				#surface_code.hz[rows_to_be_deleted_z[i]] = (surface_code.hz[rows_to_be_deleted_z[i]] + saved_z) %2
				surface_code.hz[rows_to_be_deleted_z[i]] = 0
			else:
				saved_x = surface_code.hx[rows_to_be_deleted_x[i]].copy()
				surface_code.hx[rows_to_be_deleted_x[i]] = 0
				saved_z = surface_code.hz[rows_to_be_deleted_z[i]].copy()
				surface_code.hz[rows_to_be_deleted_z[i]] = 0
		#Delte the qubit
		surface_code.hx = np.delete(surface_code.hx,defects,axis=1)	
		surface_code.hz = np.delete(surface_code.hz,defects,axis=1)			   
		surface_code=css_code(surface_code.hx,surface_code.hz)

		#surface_code.hx[surface_code.hx[:,random_integer]== 1] = 0
		#surface_code.hz[surface_code.hz[:,random_integer]== 1] = 0
		print("This broken surface code has these properties")
		if surface_code.test():
			Calculate_Distance(surface_code)
		print("My gauge fix")
		surface_code.lx = lx_0
		surface_code.lz = lz_0
		surface_code.K = 2
		# Block check 
		if surface_code.test():
			Calculate_Distance(surface_code)
		else:
			print("Check your numbers")
		#lk = css_decode_sim(hx=damagedBikeCode_One.hx, hz=damagedBikeCode_One.hz, **osd_options)
		# Surface Code test




