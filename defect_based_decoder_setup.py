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
import itertools
from ldpc import bposd_decoder
from bposd.css import css_code
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from itertools import combinations
# This has the parity matrices(re-usable)
from defect_parity_generator import *
# Use the same defect parity 
# Step 1 : Create and Extract Parity and Connection Matrices
# Step 2: Create a modified connection Schedule
# Step 3: Simulate

#######
def node_gen(A1,A2,A3,B1,B2,B3,n2=72,defects=[]):
	# Annotate the dead data qubit
	dead_data_loc = []
	for locations in defects:
		loc_flag = locations // n2
		if loc_flag:
			dead_data_loc.append(('data_right',locations % n2))
		else:
			dead_data_loc.append(('data_left',locations % n2))    
	lin_order = {}
	data_qubits = []
	Xchecks = []
	Zchecks = []
	check_defect_x_locations = []
	cnt = 0
	for i in range(n2):
		node_name = ('Xcheck', i)
		if i not in check_defect_x_locations:
			Xchecks.append(node_name)
			lin_order[node_name] = cnt
			cnt += 1

	for i in range(n2):
		node_name = ('data_left', i)
		if 1:
			data_qubits.append(node_name)
			lin_order[node_name] = cnt
			cnt += 1
		else:
			print("Dead data qubit skipped ",node_name)    
	for i in range(n2):
		node_name = ('data_right', i)
		if 1:
			data_qubits.append(node_name)
			lin_order[node_name] = cnt
			cnt += 1
		else:
			print("Dead data qubit skipped ",node_name)  



	for i in range(n2):
		node_name = ('Zcheck', i)
		Zchecks.append(node_name)
		lin_order[node_name] = cnt
		cnt += 1

	print("The effective number of physical qubits including ancillas is ",cnt)
	# compute the list of neighbors of each check qubit in the Tanner graph
	nbs = {}
	# iterate over X checks
	# Skip bad connections
	flags = []
	for i in range(n2):
		check_name = ('Xcheck',i)
		# left data qubits
		if i not in check_defect_x_locations:
			X1 = ('data_left',np.nonzero(A1[i,:])[0][0])
			
			X2 = ('data_left',np.nonzero(A2[i,:])[0][0])
			
			X3 = ('data_left',np.nonzero(A3[i,:])[0][0])
			
			X4 = ('data_right',np.nonzero(B1[i,:])[0][0])
			
			X5 = ('data_right',np.nonzero(B2[i,:])[0][0])
			
			X6 = ('data_right',np.nonzero(B3[i,:])[0][0])
			
			all_conn_x = [X1,X2,X3,X4,X5,X6]
			if len(set(all_conn_x).intersection(dead_data_loc)):
				flags.append(check_name)
				print("Check affected ",check_name)
				print("Overap at ",set(all_conn_x).intersection(dead_data_loc))
			####
			if X1 in dead_data_loc:
				X1 = None
			if X2 in dead_data_loc:
				X2 = None
			if X3 in dead_data_loc:
				X3 = None
			if X4 in dead_data_loc:
				X4 = None
			if X5 in dead_data_loc:
				X5 = None
			if X6 in dead_data_loc:
				X6 = None    





			####        
			nbs[(check_name,0)] = X1
			nbs[(check_name,1)] = X2
			nbs[(check_name,2)] = X3
			# right data qubits
			nbs[(check_name,3)] = X4
			nbs[(check_name,4)] = X5
			nbs[(check_name,5)] = X6

	# iterate over Z checks
	for i in range(n2):
		check_name = ('Zcheck',i)
		Z4 = ('data_right',np.nonzero(A1[:,i])[0][0])
		Z5 = ('data_right',np.nonzero(A2[:,i])[0][0])
		Z6 = ('data_right',np.nonzero(A3[:,i])[0][0])
		Z1 = ('data_left',np.nonzero(B1[:,i])[0][0])
		Z2 = ('data_left',np.nonzero(B2[:,i])[0][0])
		Z3 = ('data_left',np.nonzero(B3[:,i])[0][0])
		
		all_conn_z = [Z1,Z2,Z3,Z4,Z5,Z6]
		if len(set(all_conn_z).intersection(dead_data_loc)):
			flags.append(check_name)
			print("Check affected ",check_name)
			print("Overap at ",set(all_conn_z).intersection(dead_data_loc))
		if Z1 in dead_data_loc:
				Z1 = None
		if Z2 in dead_data_loc:
				Z2 = None
		if Z3 in dead_data_loc:
				Z3 = None
		if Z4 in dead_data_loc:
				Z4 = None                
		if Z5 in dead_data_loc:
				Z5 = None
		if Z6 in dead_data_loc:
				Z6 = None        
		# left data qubits
		nbs[(check_name,0)] = Z1
		nbs[(check_name,1)] = Z2
		nbs[(check_name,2)] = Z3
		# right data qubits
		nbs[(check_name,3)] = Z4
		nbs[(check_name,4)] = Z5
		nbs[(check_name,5)] = Z6
	data_qubits.remove(dead_data_loc[0])
	#TODO : Flag the affected qubits    
	print("Bad checks are ",flags)
	print("Total number of bad checks are ",len(flags))
	return (Xchecks,Zchecks,data_qubits,nbs,flags,lin_order,dead_data_loc)

def circuit_gen(args,n=288,num_cycles=12,symmetry = False):
	SKIP_ONE = True
	Xchecks,Zchecks,data_qubits,nbs,flags,lin_order,damaged = args
	sX= ['idle', 1, 4, 3, 5, 0, 2]
	sZ= [3, 5, 0, 1, 2, 4, 'idle']
	check_defect_x_locations = []
	Xbad = []
	Zbad = []
	for broken_checks in flags:
		if broken_checks[0] == 'Xcheck':
			Xbad.append(broken_checks)
		else:
			Zbad.append(broken_checks)   
	marked = []
	if SKIP_ONE:
		marked = [Zbad[2]]		 
	# syndrome measurement cycle as a list of operations
	cycle = [] 
	U = np.identity(n,dtype=int)
	# round 0: prep xchecks, CNOT zchecks and data
	t=0
	for q in Xchecks:
		if q[1] not in check_defect_x_locations:
			cycle.append(('PrepX',q))
	data_qubits_cnoted_in_this_round = []
	assert(not(sZ[t]=='idle'))
	for target in Zchecks:
		if target in marked:
				continue
		direction = sZ[t]
		control = nbs[(target,direction)]
		if control is None:
			continue
		#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
		data_qubits_cnoted_in_this_round.append(control)
		cycle.append(('CNOT',control,target))
	for q in data_qubits:
		if q in damaged:
			cycle.append(('dmg',q))
		else:
			if not(q in data_qubits_cnoted_in_this_round):
				cycle.append(('IDLE',q))

	# round 1-5: CNOT xchecks and data, CNOT zchecks and data
	for t in range(1,6):
		assert(not(sX[t]=='idle'))
		for control in Xchecks:
			if control[1] not in check_defect_x_locations:
				direction = sX[t]
				target = nbs[(control,direction)]
				if target is None:
					continue
				#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
				cycle.append(('CNOT',control,target))
		assert(not(sZ[t]=='idle'))
		for target in Zchecks:
			if target in marked:
				continue
			direction = sZ[t]
			control = nbs[(target,direction)]
			if control is None:
				continue
			#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
			cycle.append(('CNOT',control,target))
	#TODO: Superstabilizers are measured here
	#round 6(repair)- Zchecks are ready to bec checked - Xhcehcks end at round 7 
	# Look at broken X checks 
   
	print("Broken Z checks are ",Zbad)
	if (len(Zbad)!=0):
		# First Run Keep the Check dimensionality the same i.e. AB,BC,CA
		
		ZA = Zbad[0]
		ZB = Zbad[1]
		ZC = Zbad[2]
		# By default fix Z only 
		#Sweep direction from 0 to 5
		#ZAC 
		target = ZA
		for t in range(6):
			direction = t
			control = nbs[(ZC,direction)]
			if control is None:
				continue
			#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
			cycle.append(('CNOT',control,target))
		# ZBC
		target = ZB
		for t in range(6):
			direction = t
			control = nbs[(ZC,direction)]
			if control is None:
				continue
			#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
			cycle.append(('CNOT',control,target))
		# ZCA
		if not SKIP_ONE:
			target = ZC
			for t in range(6):
				direction = t
				control = nbs[(ZA,direction)]
				if control is None:
					continue
				#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
				cycle.append(('CNOT',control,target))
		
			


	# round 6: CNOT xchecks and data, measure Z checks
	t=6
	for q in Zchecks:
		if 0:
			cycle.append("badM",q)

		else:
			cycle.append(('MeasZ',q))
	assert(not(sX[t]=='idle'))
	data_qubits_cnoted_in_this_round = []
	for control in Xchecks:
		if control[1] not in check_defect_x_locations:
			direction = sX[t]
			target = nbs[(control,direction)]
			if target is None:
				continue
			U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
			cycle.append(('CNOT',control,target))
			data_qubits_cnoted_in_this_round.append(target)
	for q in data_qubits:
		if q in damaged:
			cycle.append(('dmg',q))
		else:
			if not(q in data_qubits_cnoted_in_this_round):
				cycle.append(('IDLE',q))
	# X are repaired
	print("Broken X checks are ",Xbad)

	if len(Xbad)!=0:
		if symmetry:
			XA = Xbad[0]
			XB = Xbad[1]
			XC = Xbad[2]
			#Sweep direction from 0 to 5
			#XAB 
			control = XA
			for t in range(6):
				direction = t
				target = nbs[(XB,direction)]
				if control is None:
					continue
				#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
				cycle.append(('CNOT',control,target))
			#XBC
			control = XB
			for t in range(6):
				direction = t
				target = nbs[(XC,direction)]
				if control is None:
					continue
				#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
				cycle.append(('CNOT',control,target))
			#XCA
			control = XC
			for t in range(6):
				direction = t
				target = nbs[(XB,direction)]
				if control is None:
					continue
				#U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
				cycle.append(('CNOT',control,target))    
				




	# round 7: all data qubits are idle, Prep Z checks, Meas X checks
	for q in data_qubits:
		if q in damaged:
			cycle.append(('dmg',q))
			#continue
		else:
			cycle.append(('IDLE',q))
	for q in Xchecks:
		if q[1] not in check_defect_x_locations:
			cycle.append(('MeasX',q))
	for q in Zchecks:
		cycle.append(('PrepZ',q))

	# full syndrome measurement circuit
	cycle_repeated = num_cycles*cycle
	#print(cycle_repeated)
	return cycle_repeated,cycle

def noisy_history_creator(fullcycle,err_rate= 0.001):
	error_rate = err_rate
	error_rate_init = error_rate
	error_rate_idle = error_rate
	error_rate_cnot = error_rate
	error_rate_meas = error_rate
	error_damage = 0
	print('error rate=',error_rate)
	print('Generating noisy circuits with a singe Z-type faulty operation...')
	ProbZ = []
	circuitsZ = []
	head = []
	tail = fullcycle.copy()
	for gate in fullcycle:
		assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ',"badM","badInit","dmg"])
		if gate[0]=='MeasX':
			assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(error_rate_meas)
		if gate[0]=='badM':
			#assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(0)	
		# move the gate from tail to head
		head.append(gate)
		tail.pop(0)
		assert(fullcycle==(head+tail))
		if gate[0]=='badInit':
			#assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(0)
		if gate[0]=='PrepX':
			assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(error_rate_init)
		if gate[0]=='IDLE':
			assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(error_rate_idle*2/3)
		if gate[0] == 'dmg':
			assert(len(gate)==2)
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			#ProbZ.append(error_damage)
				 
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			# add error on the control qubit
			circuitsZ.append(head + [('Z',gate[1])] + tail)
			ProbZ.append(error_rate_cnot*4/15)
			# add error on the target qubit
			circuitsZ.append(head + [('Z',gate[2])] + tail)
			ProbZ.append(error_rate_cnot*4/15)
			# add ZZ error on the control and the target qubits
			circuitsZ.append(head + [('ZZ',gate[1],gate[2])] + tail)
			ProbZ.append(error_rate_cnot*4/15)
			
	num_errZ=len(circuitsZ)
	print('Number of noisy circuits=',num_errZ)
	print('Done.')
	print('Generating noisy circuits with a singe X-type faulty operation...')
	ProbX = []
	circuitsX = []
	head = []
	tail = fullcycle.copy()
	for gate in fullcycle:
		assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ',"badM","badInit","dmg"])
		if gate[0]=='MeasZ':
			assert(len(gate)==2)
			circuitsX.append(head + [('X',gate[1])] + tail)
			ProbX.append(error_rate_meas)
		if gate[0]=='badM':
			#assert(len(gate)==2)
			circuitsX.append(head + [('X',gate[1])] + tail)
			ProbX.append(0)	
		if gate[0] == 'dmg':
			circuitsX.append(head + [('X',gate[1])] + tail)
			#ProbX.append(0)	    
		# move the gate from tail to head
		head.append(gate)
		tail.pop(0)
		assert(fullcycle==(head+tail))
		if gate[0]=='PrepZ':
			assert(len(gate)==2)
			circuitsX.append(head + [('X',gate[1])] + tail)
			ProbX.append(error_rate_init)
		if gate[0]=='IDLE':
			assert(len(gate)==2)
			circuitsX.append(head + [('X',gate[1])] + tail)
			ProbX.append(error_rate_idle*2/3)
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			# add error on the control qubit
			circuitsX.append(head + [('X',gate[1])] + tail)
			ProbX.append(error_rate_cnot*4/15)
			# add error on the target qubit
			circuitsX.append(head + [('X',gate[2])] + tail)
			ProbX.append(error_rate_cnot*4/15)
			# add XX error on the control and the target qubits
			circuitsX.append(head + [('XX',gate[1],gate[2])] + tail)
			ProbX.append(error_rate_cnot*4/15)
			
		
	num_errX=len(circuitsX)
	print('Number of noisy circuits=',num_errX)
	print('Done.')
	return circuitsX,ProbX,circuitsZ,ProbZ

# we only look at the action of the circuit on Z errors; 0 means no error, 1 means error
def simulate_circuitZ(C,lin_order,n=288):
	syndrome_history = []
	# keys = Xchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			state[control] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepX':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasX':
			assert(len(gate)==2)
			assert(gate[1][0]=='Xcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['Z','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX', 'YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ','XY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZZ','YY','YZ','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt
# we only look at the action of the circuit on X errors; 0 means no error, 1 means error
def simulate_circuitX(C,lin_order,n=288):
	syndrome_history = []
	# keys = Zchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		#print(gate)
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			#print("Gate name is ",gate)
			control = lin_order[gate[1]]
			#control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			#print("Control is ",control)
			#print("")
			state[target] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepZ':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasZ':
			assert(len(gate)==2)
			assert(gate[1][0]=='Zcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['X','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ', 'YZ']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XX','YY','XY','YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt

def simplified_parity_matrices(CX,PX,CZ,PZ,lin_order,cycle,data_qubits,Zchecks,Xchecks,LX,LZ,n2=72,num_cycles=12,k=12):
	HXdict  = {}
	circuitsX = CX.copy()
	ProbX = PX.copy()
	lx = LX.copy()
	lz = LZ.copy()
	circuitsZ = CZ.copy()
	ProbZ = PZ.copy() 
	# execute each noisy circuit and compute the syndrome
	# we add two noiseless syndrome cycles at the end
	print('Computing syndrome histories for single-X-type-fault circuits...')
	cnt = 0
	for circ in circuitsX:
		syndrome_history,state,syndrome_map,err_cnt = simulate_circuitX(circ+cycle+cycle,lin_order)
		#Disable pesky asserts
		#print("syndrome history length is ",len(syndrome_history))
		#print("to equal ",n2*(num_cycles+2))
		assert(err_cnt==1)
		assert(len(syndrome_history)==n2*(num_cycles+2))

		state_data_qubits = [state[lin_order[q]] for q in data_qubits]
		#print("data qubiyts are ",data_qubits)
		#print("state data qubits weight ",state_data_qubits)
		#x = state_data_qubits.shape
		syndrome_final_logical = (lz @ state_data_qubits) % 2
		# apply syndrome sparsification map
		syndrome_history_copy = syndrome_history.copy()
		for c in Zchecks:
			pos = syndrome_map[c]
			assert(len(pos)==(num_cycles+2))
			for row in range(1,num_cycles+2):
				syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
		syndrome_history%= 2
		syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
		supp = tuple(np.nonzero(syndrome_history_augmented)[0])
		if supp in HXdict:
			HXdict[supp].append(cnt)
		else:
			HXdict[supp]=[cnt]
		cnt+=1
		
	#n2 = n2-1
	first_logical_rowX = (n2)*(num_cycles+2) 
	print('Done.')

	# if a subset of columns of H are equal, retain only one of these columns
	print('Computing effective noise model for the X-decoder...')

	num_errX = len(HXdict)
	print('Number of distinct X-syndrome histories=',num_errX)
	HX = []
	HdecX = []
	channel_probsX = []
	for supp in HXdict:
		new_column = np.zeros((n2*(num_cycles+2)+k,1),dtype=int)
		new_column_short = np.zeros((n2*(num_cycles+2),1),dtype=int)
		new_column[list(supp),0] = 1
		new_column_short[:,0] = new_column[0:first_logical_rowX,0]
		HX.append(coo_matrix(new_column))
		HdecX.append(coo_matrix(new_column_short))
		channel_probsX.append(np.sum([ProbX[i] for i in HXdict[supp]]))
	print('Done.')
	HX = hstack(HX)
	HdecX = hstack(HdecX)

	print('Decoding matrix HX sparseness:')
	print('max col weight=',np.max(np.sum(HdecX,0)))
	print('max row weight=',np.max(np.sum(HdecX,1)))

	#n2 += 1
	HZdict  = {}

	print('Computing syndrome histories for single-Z-type-fault circuits...')
	cnt = 0
	for circ in circuitsZ:
		syndrome_history,state,syndrome_map,err_cnt = simulate_circuitZ(circ+cycle+cycle,lin_order)
		assert(err_cnt==1)
		assert(len(syndrome_history)==n2*(num_cycles+2))
		state_data_qubits = [state[lin_order[q]] for q in data_qubits]
		syndrome_final_logical = (lx @ state_data_qubits) % 2
		# apply syndrome sparsification map
		syndrome_history_copy = syndrome_history.copy()
		for c in Xchecks:
			pos = syndrome_map[c]
			assert(len(pos)==(num_cycles+2))
			for row in range(1,num_cycles+2):
				syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
		syndrome_history%= 2
		syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
		supp = tuple(np.nonzero(syndrome_history_augmented)[0])
		if supp in HZdict:
			HZdict[supp].append(cnt)
		else:
			HZdict[supp]=[cnt]
		cnt+=1


	first_logical_rowZ = n2*(num_cycles+2)
	print('Done.')

	# if a subset of columns of HZ are equal, retain only one of these columns
	print('Computing effective noise model for the Z-decoder...')
	num_errZ = len(HZdict)
	print('Number of distinct Z-syndrome histories=',num_errZ)
	HZ = []
	HdecZ = []
	channel_probsZ = []
	for supp in HZdict:
		new_column = np.zeros((n2*(num_cycles+2)+k,1),dtype=int)
		new_column_short = np.zeros((n2*(num_cycles+2),1),dtype=int)
		new_column[list(supp),0] = 1
		new_column_short[:,0] = new_column[0:first_logical_rowZ,0]
		HZ.append(coo_matrix(new_column))
		HdecZ.append(coo_matrix(new_column_short))
		channel_probsZ.append(np.sum([ProbZ[i] for i in HZdict[supp]]))
	print('Done.')
	HZ = hstack(HZ)
	HdecZ = hstack(HdecZ)


	print('Decoding matrix HZ sparseness:')
	print('max col weight=',np.max(np.sum(HdecZ,0)))
	print('max row weight=',np.max(np.sum(HdecZ,1)))

	return HX,HdecX,HZ,HdecZ,first_logical_rowX,first_logical_rowZ,channel_probsX,channel_probsZ




########




if __name__ == "__main__":
	num_trials = 10000
	args = sys.argv[1:]
	if args[0] == "-err":
		perr = float(args[1])
	if args[2] == "-repair":
		repair = args[3]
	damageQubits = [0] # Hard-Code
	# Parity Matrices Extracted
	codeName = code_dict(name="gross")
	bikeCode = bivariate_parity_generator_bicycle(codeName)
	#ncyc = 12
	ncyc = 1
	sym = False
	if repair == "sym":
		sym = True
	bikeCode_damaged = damage_qubit(bikeCode,turnOffQubits=damageQubits,symmetry=sym)
	k = bikeCode_damaged.K
	hx = bikeCode_damaged.hx.copy()
	hz = bikeCode_damaged.hz.copy()
	lx = bikeCode_damaged.lx
	lz = bikeCode_damaged.lz
	# Extract the connection matrices
	A1,A2,A3,B1,B2,B3 = connection_matrices(code_dict(name="gross"))
	parameters = node_gen(A1,A2,A3,B1,B2,B3,defects=damageQubits)
	circuit,cycle = circuit_gen(parameters,num_cycles=ncyc,symmetry=sym) # With defect-reapir added
	# Generate noisy circuits
	circuitX,pX,circuitZ,pZ = noisy_history_creator(circuit,err_rate=perr)
	linearorder = parameters[-2]
	Xcheck = parameters[0]
	Zcheck = parameters[1]
	data = parameters[2]
	Hx,Hdx,Hz,HdZ,fX,fZ,channel_probsX,channel_probsZ = simplified_parity_matrices(circuitX,pX,circuitZ,pZ,linearorder,cycle,data,Zcheck,Xcheck,LZ=lz,LX=lx,num_cycles=ncyc)
	# Save Stuff
	# save decoding matrices 
	(ell,m,a1,a2,a3,b1,b2,b3) = codeName
	sX= ['idle', 1, 4, 3, 5, 0, 2]
	sZ= [3, 5, 0, 1, 2, 4, 'idle']
	mydata = {}
	mydata['HdecX']=Hdx
	mydata['HdecZ']=HdZ
	mydata['probX']=channel_probsX
	mydata['probZ']=channel_probsZ
	mydata['cycle']=cycle
	mydata['lin_order']=linearorder
	mydata['num_cycles']=ncyc
	mydata['data_qubits']=data
	mydata['Xchecks']=Xcheck
	mydata['Zchecks']=Zcheck
	mydata['HX']=Hx
	mydata['HZ']=Hz
	mydata['lx']=lx
	mydata['lz']=lz
	mydata['first_logical_rowZ']=fZ
	mydata['first_logical_rowX']=fX
	mydata['ell']=ell
	mydata['m']=m
	mydata['a1']=a1
	mydata['a2']=a2
	mydata['a3']=a3
	mydata['b1']=b1
	mydata['b2']=b2
	mydata['b3']=b3
	mydata['error_rate']=perr
	mydata['sX']=sX
	mydata['sZ']=sZ
	n = 144
	k =12
	error_rate = perr
	num_cycles = ncyc
	if sym:
		title='./TMP/symmetric_mydata_' + str(n) + '_' + str(k) + '_p_' + str(error_rate) + '_cycles_' + str(num_cycles)
	else:
		title = './TMP/asymmetric_mydata_final_' + str(n) + '_' + str(k) + '_p_' + str(error_rate) + '_cycles_' + str(num_cycles)

	print('saving data to ',title)
	with open(title, 'wb') as fp:
		pickle.dump(mydata, fp)

	print('Done')

