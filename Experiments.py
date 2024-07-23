import os
import sys
from code_capacity_simulation_setup import code_capacity_simulation
# Tell the name of the code as arg
args = sys.argv[1:]
if args[0] == "-code":
    codeName = args[1]
path = "Results//CodeCapacity//PaperResults//"
#fn = codeName+"final_exceptdistant_finaltake.json"
fn = codeName+"finalPaper.json"
fileName = path+fn
#options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':1000,'fileName':fileName,'progressTrack':True,'defectModes':['common'],'repairModes':['ZX']}
#options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':100000,'fileName':fileName,'progressTrack':True,'defectModes':['common','onlyx','onlyz']}
options = {'defectType':'doubleDefect','codeName':codeName,'nTrials':100000,'fileName':fileName,'progressTrack':True,'R_mode':False}
zero = code_capacity_simulation(**options)
#v2 simulations
#options = {'defectType':'singledefect','codeName':codeName,'nTrials':100000,'fileName':fileName,'progressTrack':True,'R_mode':True}
#zero = code_capacity_simulation(**options)