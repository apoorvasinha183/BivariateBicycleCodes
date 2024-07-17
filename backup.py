import os
import sys
from circuit_simulation_setup import circuit_sims
from defect_parity_generator import *
# Tell the name of the code as arg
args = sys.argv[1:]
if args[0] == "-code":
    codeName = args[1]
nyc = ncycles(name= codeName)
#nyc = 1
path = "./Results//CircuitSims//"
fileName = path
#options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':1000,'fileName':fileName,'progressTrack':True,'defectModes':['common'],'repairModes':['ZX']}
#options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':100000,'fileName':fileName,'progressTrack':True,'defectModes':['common','onlyx','onlyz']}
options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':1000,'fileName':fileName,'progressTrack':True,'ncyc':nyc,'perrors':[0.005,0.006,0.007]}
Ckt = circuit_sims(**options)
#Ckt.generate_detector_models()
Ckt.decode_all()