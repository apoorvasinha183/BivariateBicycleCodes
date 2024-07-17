# Master class definition for organizing all Code-Capacity simulations : Setup similar to Roffee's code
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


# Class definition
class code_capacity_simulation():
    """
    Class to organize and conduct code capacity simulations. 
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

    """
    def __init__(self, **input_dict):
        self.qcode = None
        self.N = 1e9
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
            'R_mode':False,
            'perrors':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        }
        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]
        #self.perrors = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        self.output_file = self.__dict__['fileName']
        self.output = {}
        self.verbose = True
        #self.defectModes = []
        #self.repairModes = []
        self.errors = ['z','x']
        self.etype = 'z'
        # Debug message
        self._status_message(status="start")
        # Construct the undamaged code
        self._undamaged_code()
        # Setup output
        self.setup_output()
        
        # Simulate
        self.simulate()
        #Wrap Up
        self._status_message(status="end")
    def setup_output(self):
        self.output["N"] = self.N
        self.output["K"] = self.K
        self.output["nTrials"] = self.__dict__['nTrials']
        self.output["nBad"] = 0
        self.output["repairType"] = self.__dict__['repairType']
        self.output["defectType"] = self.__dict__['defectType']
        self.output["defectLocation"] = self.__dict__['defectLocation']
        self.output["ler"] = 0.00
        self.output["perr"] = 0.00
        self.output["d"] = 50
        self.output["d"] = 50
        self.output["type"] = 'z'
        self.output['time'] = self.starttime.isoformat()
        self.output['repair_version'] = 'v2' if self.R_mode else 'v1'
        return

    def _undamaged_code(self):
        self.qcode = bivariate_parity_generator_bicycle(codeParams=code_dict(name = self.__dict__['codeName']))
        self.N = self.qcode.N
        self.K = self.qcode.K
        self.qcode = [self.qcode]
        return
    def _status_message(self,status="start"):
        if self.__dict__['logging']:
            if status == "start":
                self.starttime = datetime.now()
                print("These sims were fired on ",self.starttime)
                if self.verbose:
                    print("Printing the simulation settings below for you to debug \n")
                    for keys in self.__dict__:
                        print(f"Input:{keys};Value:{self.__dict__[keys]}")
            else:
                self.endtime = datetime.now()
                print(f"Simulation finished. Took {self.endtime-self.starttime}")        
        return        
    def _damaged_code(self,qcode=None,repair='A-Z'):
        if repair is None:
            return
        if repair == 'A-Z':
            if self.R_mode:
                self.qcode = [damage_qubit_v2(qcode,turnOffQubits=self.__dict__['defectLocation'])]
            else:
                self.qcode = [damage_qubit(qcode,turnOffQubits=self.__dict__['defectLocation'])]
            return
        if repair == 'A-X':
            if not self.R_mode:
                self.qcode = [damage_qubit(qcode,turnOffQubits=self.__dict__['defectLocation'],flip=True)]
            else:
                self.qcode = [damage_qubit_v2(qcode,turnOffQubits=self.__dict__['defectLocation'],flip=True)]
            return
        if repair == 'symmetry':
            if not self.R_mode:
                self.qcode = [damage_qubit(qcode,turnOffQubits=self.__dict__['defectLocation'],symmetry=True)]
            else:
                self.qcode = [damage_qubit_v2(qcode,turnOffQubits=self.__dict__['defectLocation'],symmetry=True)]
            return
        if repair == 'ZZ':
            if not self. R_mode:
                self.qcode = [damage_qubit(qcode,turnOffQubits=self.deadQubits,test=self.verbose)]
            else:
                self.qcode = [damage_qubit_v2(qcode,turnOffQubits=self.deadQubits,test=self.verbose)]
            return
        if repair == 'ZX':
            if not self. R_mode:
                self.qcode = [damage_qubit(qcode,turnOffQubits=self.deadQubits,alterning=True,test=self.verbose)]
            else:
                self.qcode = [damage_qubit_v2(qcode,turnOffQubits=self.deadQubits,alterning=True,test=self.verbose)]
            return
        return
    def _processdefectMode(self,mode=None):
        # Randomly sample a qubit from the respective set
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
        #TODO : Delete comments from everywhere
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




    def simulate(self):
        #TODO : Add options for defect modes
        # Generate all possible defect modes
        self.error_modes()
        code_original = self.qcode.copy()
        for modes in self.defectModes:
            self.output["defectType"] = modes
            if modes != "singledefect":
                self._divideQubits(code_original)
                self._processdefectMode(modes)
                #sys.exit()
                
            for repairs in self.repairModes:
                self.output["repairType"] = repairs
                #Make a copy of the 
                
                for codes in code_original:
                    self._damaged_code(qcode=codes,repair=repairs)
                    for pauli in self.errors:
                        self.etype = pauli
                        self.output["d"] = 50
                        self.output["d"] = 50
                        for errors in self.perrors:
                            self.output["perr"] = errors
                            self.simulate_per_error(code=self.qcode[0],error=errors,type=self.etype)
                            if self.output_file!=None:
                                f=open(self.output_file,"a")
                                f.write(json.dumps(self.output) + "\n")
                                f.close()
                        print(f"MC Distance for repair: {repairs} on defect type: {modes} is {self.output['d']} for code: {self.__dict__['codeName']} for Pauli Error {pauli}")

        return 0
        #pass
    def error_modes(self):
        if self.__dict__['defectType'] is None:
            self.repairModes =[None]
            self.defectModes=[None]
        if self.__dict__['defectType'] == 'singledefect':
            self.repairModes =['A-X','A-Z','symmetry']
            self.defectModes=['singledefect']
        if self.__dict__['defectType'] == 'doubleDefect':
            print("Defect Modes are ",self.defectModes)
            print("Repair Modes are ",self.repairModes)
            if self.repairModes is  None:
                self.repairModes = ['ZZ','ZX']
            if self.defectModes is  None:
                self.defectModes = ['distant','common','onlyx','onlyz']    

    def simulate_per_error(self,code=None,error=0.01,type='z'):
        if code is None:
            code  = self.qcode[0]
        faultRate,minWt,e=Sample_Distance(code,perr=error,nTrials=self.__dict__['nTrials'],type=type,disable=not self.__dict__['progressTrack'])
        self.output["ler"] = float(faultRate)
        self.output["type"] =self.etype
        if self.etype == 'z':
            self.output["d"] = int(min(minWt,self.output["d"]))
            
        else:
            self.output["d"] = int(min(minWt,self.output["d"]))

        self.output["nBad"] = int(e)
        return



