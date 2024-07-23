# Vary defects from 0.1% to 5% and see the effective distance.
# A-XX vs A-ZZ vs Symmteric vs Symmetric-Gauged vs XZ vs ZX repairs
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
from explicit_distance import *
import random
import matplotlib.pyplot as plt
nTrials = 1000
pdef = [0.001,0.005,0.01,0.05]
#Helper 
def choose_sites(N, p):
	chosen_numbers = []
	for number in range(N):
		if random.random() < p:
			chosen_numbers.append(number)
	return chosen_numbers


Name = 'd10-90'
codeName = code_dict(name=Name)
bikeCode = bivariate_parity_generator_bicycle(codeName)
N1 = bikeCode.N
N = len(pdef)
d_xx_F = []
d_zz_F = []
d_sym_F =[]
d_symg_F = []
d_zx_F = []
d_xz_F = []
for defect_prob in tqdm(pdef):
	# Sample a defect string nTrial times , loop it
	d_xx = np.full(nTrials,np.nan)
	d_zz = np.full(nTrials,np.nan)
	d_sym = np.full(nTrials,np.nan)
	d_symg = np.full(nTrials,np.nan)
	d_zx = np.full(nTrials,np.nan)
	d_xz = np.full(nTrials,np.nan)
	for i in tqdm(range(nTrials)):
		#sample an error string
		defects = choose_sites(N1,defect_prob)
		if len(defects) == 0:
			full_dist = 10 #hARD-code for now
			d_xx[i] = full_dist
			d_zz[i] = full_dist
			d_sym[i] = full_dist
			d_symg[i] = full_dist
			d_zx[i] = full_dist
			d_xz[i] = full_dist
			continue
		else:
			# Damage the code
			#A-XX repair
			damagedBikeCode = damage_qubit(bikeCode,turnOffQubits=defects,symmetry=False,flip=True,test=True)
			# Save the logicals for gauging later
			lx1 = damagedBikeCode.lx.copy() # This one
			lz1 = damagedBikeCode.lz.copy()
			# Sample the distance the monte-carlo way
			_,d_xx[i],_ =Sample_Distance(damagedBikeCode,type='x',perr=0.04,nTrials = 1000,disable= False)
			#A-ZZ repair
			damagedBikeCode = damage_qubit(bikeCode,turnOffQubits=defects,symmetry=False,flip=False,test=True)
			# Save the logicals for gauging later
			lx2 = damagedBikeCode.lx.copy() 
			lz2 = damagedBikeCode.lz.copy() # This one
			#Sample the distance the monte-carlo way
			_,d_zz[i],_ =Sample_Distance(damagedBikeCode,type='z',perr=0.04,nTrials = 1000,disable= False)
			#Symmetric repair
			damagedBikeCode = damage_qubit(bikeCode,turnOffQubits=defects,symmetry=True,test=True)
			_,d_sym[i],_ =Sample_Distance(damagedBikeCode,type='z',perr=0.04,nTrials = 1000,disable= False)
			# Transfer logicals 
			damagedBikeCode.lx = lx1
			damagedBikeCode.lz = lz2
			damagedBikeCode.K = bikeCode.K
			# Test your logic
			if damagedBikeCode.test():
				_,d_symg[i],_ =Sample_Distance(damagedBikeCode,type='z',perr=0.04,nTrials = 1000,disable= False)
			else:
				print("Something went wrong")
				d_symg[i] = 0
			#A - XZ reapir
			damagedBikeCode = damage_qubit(bikeCode,turnOffQubits=defects,symmetry=False,flip=True,alterning=True,test=True)
			_,d1,_ =Sample_Distance(damagedBikeCode,type='z',perr=0.04,nTrials = 1000,disable= False)
			_,d2,_ =Sample_Distance(damagedBikeCode,type='x',perr=0.04,nTrials = 1000,disable= False)
			d_xz[i] = min(d1,d2)
			#A-ZX repair
			damagedBikeCode = damage_qubit(bikeCode,turnOffQubits=defects,symmetry=False,flip=False,alterning=True,test=True)
			_,d1,_ =Sample_Distance(damagedBikeCode,type='z',perr=0.04,nTrials = 1000,disable= False)
			_,d2,_ =Sample_Distance(damagedBikeCode,type='x',perr=0.04,nTrials = 1000,disable= False)
			d_zx[i] = min(d1,d2)
	d_xx_F.append(np.mean(d_xx)) 
	d_zz_F.append(np.mean(d_zz)) 
	d_sym_F.append(np.mean(d_sym))
	d_symg_F.append(np.mean(d_symg))
	d_zx_F.append(np.mean(d_zx))
	d_xz_F.append(np.mean(d_xz))

# Plot
plt.plot(pdef,d_xx_F,label ='XX-repair ')
plt.plot(pdef,d_zz_F,label ='ZZ-repair ')
plt.plot(pdef,d_sym_F,label ='sym-repair ')
plt.plot(pdef,d_symg_F,label ='sym-repair(gauged)')
plt.plot(pdef,d_xz_F,label ='XZ-repair ')
plt.plot(pdef,d_zx,label ='ZX-repair ')
plt.xlabel('Defect Rate Data Error')
plt.ylabel('Average distance')
plt.legend(fontsize='large')   # Set the font size of the legend
plt.legend(title='Legend')
plt.legend(loc='lower right')  
plt.savefig("DefectRatevsDistance.png")
plt.title("Efficacy of data recovery strategies at every fault rate")
plt.show()
	



