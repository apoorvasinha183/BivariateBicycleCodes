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
	shuffled_indices = np.random.permutation(k)
	lx_shuffled = lx[shuffled_indices]
	lz_shuffled = lz[shuffled_indices]
	k =1
	pbar = tqdm(range(k),ncols=0)
	#Randomize the order of the lx and lz matrices
	for i in pbar:
		

		w1,hxSmalli = distance_test(hx,lx_shuffled[i,:])
		w2,hzSmalli = distance_test(hz,lz_shuffled[i,:])
		#print('Logical qubitX=',i,'Distance=',w1)
		#print('Logical qubitZ=',i,'Distance=',w2)
		qubitix.append(w1)
		qubitiz.append(w2)
		hxSmall[i] = hxSmalli
		hzSmall[i] =  hzSmalli
		#w1 = 10000
		d = min(d,w1,w2)
		pbar.set_description(f"d_min: {d};")
	pbar.close()
	print("X distances ",qubitix)
	print("Z distances ",qubitiz)
	print("Minimum distance is ",d)

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
	codeName = code_dict(name="gross")
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_One = damage_qubit(bikeCode)
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	damagedBikeCode_Two = damage_qubit(bikeCode,turnOffQubits=[0,12])
	errors = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
	trials = 1000
	dn = 50
	d1x = 50
	d1z = 50
	d2x = 50
	d2z = 50
	for p in errors:
		#Calculate_Distance(bikeCode)
		print("Normal Code")
		faultRate,minWt,_=Sample_Distance(bikeCode,perr=p,nTrials=trials)
		dn = min(dn,minWt)
		#Cause a data damage(default is 0)
		
		#Calculate_Distance(damagedBikeCode)
		print("1 data defect Code-AZ")
		damagedBikeCode_One = damage_qubit(bikeCode)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='z')
		d1z = min(d1z,minWt)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='x')
		d1x = min(d1x,minWt)
		#--
		print("1 data defect Code-AX")
		damagedBikeCode_One = damage_qubit(bikeCode,flip=True)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='z')
		d1z = min(d1z,minWt)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='x')
		d1x = min(d1x,minWt)
		print("1 data defect Code-Symmetric")
		damagedBikeCode_One = damage_qubit(bikeCode,symmetry=True)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='z')
		d1z = min(d1z,minWt)
		faultRate,minWt,_=Sample_Distance(damagedBikeCode_One,perr=p,nTrials=trials,type='x')
		d1x = min(d1x,minWt)
		# Cause double data damage(0,12)
		#print("2 data defects Code")
		#faultRate,minWt,_=Sample_Distance(damagedBikeCode_Two,perr=p,nTrials=trials,type='z')
		#d2z = min(d2z,minWt)
		#faultRate,minWt,_=Sample_Distance(damagedBikeCode_Two,perr=p,nTrials=trials,type='x')
		#d2x = min(d2x,minWt)
	print(f"Normal dn:{dn}")	
	print(f"1Qubit x:{d1x} z:{d1z}")
	print(f"2Qubit x:{d2x} z:{d2z}")



