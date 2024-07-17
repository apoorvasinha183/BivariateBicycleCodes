# Master Class Defintions that organize the circuit level noise simulations
#Libraries to Import
# Libraries to import
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
from explicit_distance import *
from datetime import datetime
import json
from defect_aggereate_setup import *
import os
from decoder_distance import *
from circuit_distance_all import *



# Class Definition
class circuit_sims():
	"""
	Class to organize and conduct circuit noise level simulations. 
	Parameters may be entered directly or as a dictionary with the proper keywords.
	--- Parameters ---
	codeName : string . Name of the code family . You may add your own in defect_parity_generator.py DEFAULT: "gross" ([[144,12,12]])
	nTrials : Integer . Number of Monte-Carlo Shots DEFAULT : 10000
	defectType : string . Type of the defect : 1. "singledatadefect" 2. "singleparitydefect" 3. "doubleparitydefect" 4. "doubledatadefect" 5."nodefect"6. "custom" DEFAULT : "singledatadefect"
	repairType : string . How the data repair is patched SINGLE DATA FAULT : "symmetric" ;"A-X" ;"A-Z" (DEFAULT : "A-Z")MULTI DATA DEFECTS : "XX";"ZZ";"ZX";"XZ" (DEFAULT : "ZZ")
	defectLocation : List[Int] Data/Parity Location where you want to insert qubit damage . DEFAULT : [0]
	fileName : Path  . Path to the file/location where you want to save the raw data. DEFAULT : None
	logging : Boolean . To enable or disable debug messages . DEFAULT : True
	progressTrack : Boolean .To enable or disable loop progress bars to plan your coffee breaks/weekend/vacation DEFAULT : True
	perrors : List[Float] Range of noise values you want to simulate DEFAULT : [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
	decode : Boolean . If you want to decode the code family  DEFAULT : False
	sample : Boolean . If you want to create the error model DEFAULT :True
	"""
	def __init__(self, **input_dict):
		self.qcode = None
		self.N = 1e9
		self.DEBUG = False
		self.DEBUG_Title = None
		# Default Inputs
		default_input = {
			'codeName':"gross",
			'nTrials':10000,
			'defectType':"singledefect",
			'repairType':"A-Z",
			'defectLocation':[0],
			'fileName':None,
			'logging':True,
			'progressTrack':True,
			'defectModes':None,
			'repairModes':None,
			'perrors':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
			'decode':False,
			'sample':True,
			'ncyc':12
		}
		for key in input_dict.keys():
			self.__dict__[key] = input_dict[key]
		for key in default_input.keys():
			if key not in input_dict:
				self.__dict__[key] = default_input[key]
		self.output = {}
		self.error_dict={}
		self.verbose = True
		self.repair = None
		self.situation = None
		self.e = 0
		self.orderx = []
		self.orderz = []
		self.anchors = []
		self.Xbad = []
		self.Zbad = []
		# Construct the undamaged code
		self._undamaged_code()
		self._status_message(status="start")
	def _debug(self,file):
		#decode_file(file)
		sample_distance_from_file(file)
	def _setup__output(self):
		if self.sample:
			self.output['HdecX']=None
			self.output['HdecZ']=None
			self.output['probX']=None
			self.output['probZ']=None
			self.output['cycle']=None
			self.output['lin_order']=None
			self.output['num_cycles']=None
			self.output['data_qubits']=None
			self.output['Xchecks']=None
			self.output['Zchecks']=None
			self.output['HX']=None
			self.output['HZ']=None
			self.output['lx']=None
			self.output['lz']=None
			self.output['first_logical_rowZ']=None
			self.output['first_logical_rowX']=None
			self.output['ell']=None
			self.output['m']=None
			self.output['a1']=None
			self.output['a2']=None
			self.output['a3']=None
			self.output['b1']=None
			self.output['b2']=None
			self.output['b3']=None
			self.output['error_rate']=None
			self.output['sX']=None
			self.output['sZ']=None
			self.output['damaged'] = None
			self.output['style'] = None
		if self.decode:
			self.error_dict["N"] = self.N
			self.error_dict["K"] = self.K
			self.error_dict["nTrials"] = self.nTrials
			self.error_dict["nBad"] = 0
			self.error_dict["repairType"] = None
			self.error_dict["defectType"] = None
			self.error_dict["defectLocation"] = self.defectLocation
			self.error_dict["ler"] = 0.00
			self.error_dict["perr"] = 0.00
			self.error_dict['time'] = self.starttime.isoformat()
		return
	def _undamaged_code(self):
		self.qcode = bivariate_parity_generator_bicycle(codeParams=code_dict(name = self.__dict__['codeName']))
		self.N = self.qcode.N
		self.K = self.qcode.K
		self.qcode = [self.qcode]
		self.qcode_undamaged = self.qcode[0]
		return
	def generate_detector_models(self):
		#Turn on sample mode and explictly disable the decoder for now.
		self.decode = False
		self.sample = True
		self._setup__output()
		# Debug message
		self._status_message(status="start")
		self.error_modes()
		code_original = [self.qcode_undamaged]
		#self.ncyc = 12 # TODO: Handle it case by case
		for modes in self.defectModes:
			x = 1
			self.output["defectType"] = modes
			if modes != "singledefect":
				self._divideQubits(code_original)
			self._processdefectMode(modes)
			for repairs in self.repairModes:
				self.output["repairType"] = repairs
				#Make a copy of the 
				code_cap = code_original.copy()
				for codes in code_cap:
					self._damaged_code(qcode=codes,repair=repairs)
					for errors in self.perrors:
						print(f"Trying mode:{modes} and repairs:{repairs}")
						x =1
						# Run a single error dem extraction round
						alternate =False
						flip = False
						if repairs == 'ZX':
							alternate = True
						if repairs == 'A-X':
							flip = True
						print("codes is ",codes.N)
						print("deadQubits are ",self.deadQubits)
						self.detector_model_per_error(code = codes,error=errors,alternate=alternate,flip=flip)
						# Save the output
						self.save_output(error=errors,situation=modes,repair=repairs)
						if self.DEBUG:
							self._debug(self.DEBUG_Title)
						# Declare checkpoint
						self._status_message()
		self._status_message(status="end")
		return
	def decode_all(self):
		self.decode = True
		self.sample= False
		self._setup__output()
		# Sweep the error rates
		self.error_modes()
		self._status_message(status="start")
		self.error_modes()
		code_original = [self.qcode_undamaged]
		for modes in self.defectModes:
			self.error_dict["defectType"] = modes
			if modes != "singledefect":
				self._divideQubits(code_original) # TODO : Remember why I did this ? 
			self._processdefectMode(modes) # TODO : I was sleeping maybe 
			for repairs in self.repairModes: 
				self.error_dict["repairType"] = repairs
				for error_rates in self.perrors:
					self.error_dict["perr"] = error_rates
					self.error_dict["nBad"]=self.decode_per_error(code=None,error_rates=error_rates,situation=modes,repair=repairs)
					#Save
					self.save_output()
					# Declare checkpoint
					self._status_message()
		self._status_message(status="end")
		return
	def _setup_var(self):
		pass
	def save_output(self,error=0.01,situation='one',repair='A-Z'):
		path = self.fileName
		
		if self.sample:
			fn = self.codeName+"err"+str(int(1000*error))+situation+repair+str(self.ncyc)
			fileName = path+fn
			print('saving data to ',fileName)
			if self.DEBUG:
				self.DEBUG_Title = fileName
			# Extract the directory path
			dirName = os.path.dirname(fileName)

			# Create the directory if it doesn't exist
			os.makedirs(dirName, exist_ok=True)
			with open(fileName, 'wb') as fp:
				pickle.dump(self.output, fp)
		else:
			if self.decode:
				prefix_folder ="//Results_by_family//"
				fn = str(self.codeName)+".json"
				fName = path+prefix_folder+fn
				f=open(fName,"a")
				f.write(json.dumps(self.error_dict) + "\n")
				f.close()
		return
	def _status_message(self,status='now'):
		if self.__dict__['logging']:
			if status == "start":
				self.starttime = datetime.now()
				print("These sims were fired on ",self.starttime)
				if self.verbose:
					print("Printing the simulation settings below for you to debug \n")
					for keys in self.__dict__:
						print(f"Input:{keys};Value:{self.__dict__[keys]}")
			if status == "end":
				self.endtime = datetime.now()
				print(f"Simulation finished. Took {self.endtime-self.starttime}")  
			else:
				print(f"Simulation Checkpoint. Time of  {datetime.now()-self.starttime} has elpased since the beginning")
		return        
	def _processdefectMode(self,mode=None):
		# Randomly sample a qubit from the respective set
		self.deadQubits = self.defectLocation
		if mode == 'distant':
			qubitTwo = int(np.random.choice(self.distant))
			self.deadQubits = [self.__dict__['defectLocation'][0],qubitTwo]
			self.output["defectLocation"] = self.deadQubits
			return
		if mode == "common":
			qubitTwo = int(np.random.choice(self.common))
			self.deadQubits = [self.__dict__['defectLocation'][0],qubitTwo]
			self.output["defectLocation"] = self.deadQubits
			return
		if mode == "onlyx":
			qubitTwo = int(np.random.choice(self.onlyX))
			self.deadQubits = [self.__dict__['defectLocation'][0],qubitTwo]
			self.output["defectLocation"] = self.deadQubits
			return
		if mode == "onlyz":
			qubitTwo = int(np.random.choice(self.onlyZ))
			self.deadQubits = [self.__dict__['defectLocation'][0],qubitTwo]
			self.output["defectLocation"] = self.deadQubits
			return
		
	def _divideQubits(self,qcode=None):
		assert(len(self.__dict__['defectLocation'])==1)
		firstQubit = self.__dict__['defectLocation'][0]
		hx = qcode[0].hx
		hz = qcode[0].hz
		#Identify the checks which are connected to the firstQubit in the undamaged code
		broken_rows_x = hx[hx[:,firstQubit] == 1]
		broken_rows_z = hz[hz[:,firstQubit] == 1]
		defectNum,_ = broken_rows_x.shape
		print("Defective Positions in X/Z")
		qubitX = []
		qubitZ = []
		for i in range(defectNum):
			affectedRowx = broken_rows_x[i]
			affectedRowz = broken_rows_z[i]
			stab_X_read = np.where(affectedRowx == 1)[0]
			stab_Z_read = np.where(affectedRowz == 1)[0]
			qubitX = qubitX + list(stab_X_read)
			qubitZ = qubitZ+list(stab_Z_read)
			print("StabX is ",stab_X_read)
			print("StabZ is ",stab_Z_read)
		print("qubitx ",qubitX)
		print("qubitz ",qubitZ)
		# Set theory 
		all_qubits = list(np.arange(qcode[0].N))
		distant_qubits = list(set(all_qubits)-(set(qubitX)|set(qubitZ)))
		print("Distant qubits ",distant_qubits)
		common_qubits = list(set(qubitX)&set(qubitZ)-set([firstQubit]))
		print("Common qubits ",common_qubits)
		onlyX = list(set(qubitX)-set(qubitZ))
		print("OnlyX qubits ",onlyX)
		onlyZ = list(set(qubitZ)-set(qubitX))
		print("OnlyZ qubits ",onlyZ)
		self.distant = distant_qubits
		self.onlyX = onlyX
		self.onlyZ = onlyZ
		self.common = common_qubits
		return
	def detector_model_per_error(self,code = None,error=0.01,alternate=False,flip =False):
		codeName = code_dict(name=self.codeName)
		A1,A2,A3,B1,B2,B3 = connection_matrices(codeName)
		parameters = node_gen(A1,A2,A3,B1,B2,B3,defects=self.deadQubits,alternating=alternate,n2=self.N//2)
		#print(f'Damaged just before Xbad:{self.Xbad}Zbad:{self.Zbad}')
		circuit,cycle,SANITY_BIN = circuit_gen(parameters,num_cycles=self.ncyc,symmetry=False,flip=flip,alternating=alternate,n=2*self.N,anchors=self.anchors,Xbad= self.Xbad,Zbad=self.Zbad,)
		style = 'zx' if alternate else 'zz'
		# Generate noisy circuits
		circuitX,pX,circuitZ,pZ = noisy_history_creator(circuit,err_rate=error)
		linearorder = parameters[-2]
		Xcheck1 = parameters[0]
		Zcheck1 = parameters[1]
		data = parameters[2]
		lx = self.lx.copy()
		lz = self.lz.copy()
		ncyc = self.ncyc
		Hx,Hdx,Hz,HdZ,fX,fZ,channel_probsX,channel_probsZ = simplified_parity_matrices(circuitX,pX,circuitZ,pZ,linearorder,cycle,data
																				 ,Zcheck1,Xcheck1,LZ=lz,LX=lx,num_cycles=self.ncyc,n2=self.N//2,SANITY_BIN=SANITY_BIN,k=self.qcode_undamaged.K)
		# save decoding matrices 
		(ell,m,a1,a2,a3,b1,b2,b3) = codeName
		sX= ['idle', 1, 4, 3, 5, 0, 2]
		sZ= [3, 5, 0, 1, 2, 4, 'idle']
		self.output = {}
		self.output['HdecX']=Hdx
		self.output['HdecZ']=HdZ
		self.output['probX']=channel_probsX
		self.output['probZ']=channel_probsZ
		self.output['cycle']=cycle
		self.output['lin_order']=linearorder
		self.output['num_cycles']=ncyc
		self.output['data_qubits']=data
		self.output['Xchecks']=Xcheck1
		self.output['Zchecks']=Zcheck1
		self.output['HX']=Hx
		self.output['HZ']=Hz
		self.output['lx']=lx
		self.output['lz']=lz
		self.output['first_logical_rowZ']=fZ
		self.output['first_logical_rowX']=fX
		self.output['ell']=ell
		self.output['m']=m
		self.output['a1']=a1
		self.output['a2']=a2
		self.output['a3']=a3
		self.output['b1']=b1
		self.output['b2']=b2
		self.output['b3']=b3
		self.output['error_rate']=error
		self.output['sX']=sX
		self.output['sZ']=sZ
		self.output['damaged'] = SANITY_BIN
		self.output['style'] = style
		return
	def decode_per_error(self,code = None,error_rates=0.01,situation='one',repair='A-Z'):
		path = self.fileName
		fn = self.codeName+"err"+str(int(1000*error_rates))+situation+repair+str(self.ncyc)
		fileName = path+fn
		print('Decoding from ',fileName)
		return decode_file(fileName,n=self.qcode_undamaged.N,k=self.qcode_undamaged.K,num_trials=self.nTrials,num_cycles=self.ncyc)
		#pass
	def error_modes(self):
		if self.__dict__['defectType'] is None:
			self.repairModes =[None]
			self.defectModes=[None]
		if self.__dict__['defectType'] == 'singledefect':
			#self.repairModes =['A-X','A-Z','symmetry']
			self.repairModes =['A-X','A-Z']
			#self.repairModes = ['A-X']
			self.defectModes=['singledefect']
		if self.__dict__['defectType'] == 'doubleDefect':
			print("Defect Modes are ",self.defectModes)
			print("Repair Modes are ",self.repairModes)
			if self.repairModes is  None:
				self.repairModes = ['ZZ','ZX']
			if self.defectModes is  None:
				self.defectModes = ['distant','common','onlyx','onlyz'] 
	def _damaged_code(self,qcode=None,repair='A-Z'):
		if repair is None:
			print("No defect")
			#return
		if repair == 'A-Z':
			qc,anchor,Xbad,Zbad = damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose,order=True)
			self.qcode = [qc]
			#return
		if repair == 'A-X':
			qc,anchor,Xbad,Zbad = damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose,order=True,flip=True)
			self.qcode = [qc]
			#return
		if repair == 'symmetry':
			qc,anchor,Xbad,Zbad = damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose,order=True,symmetry=True)
			self.qcode = [qc]
			#return
		if repair == 'ZZ':
			qc,anchor,Xbad,Zbad = damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose,order=True)
			self.qcode = [qc]
			#return
		if repair == 'ZX':
			qc,anchor,Xbad,Zbad = damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose,order=True,alterning=True)
			self.qcode = [qc]
			#return
		self.anchors = anchor.copy()
		self.Xbad = Xbad.copy()
		self.Zbad = Zbad.copy()
		#print(f'Damaged Xbad:{self.Xbad}Zbad:{self.Zbad}')
		self.lx = self.qcode[0].lx
		self.lz = self.qcode[0].lz
		return
