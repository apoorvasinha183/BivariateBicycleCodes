import json
import numpy as np
import matplotlib.pyplot as plt



# Data Extracted from Sims () --> Data files
perror = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.01]
Nshots_undamaged = 10000*np.ones(9,dtype=int)
Nerror_undamaged = [0,1,40,355,1830,4845,8237,9661,9997]
Nerror_onerepair = [0,1,55,426,2026,5383,8500]
Nshots_undamaged_multiple = 1000*np.ones(9,dtype=int) # Runtime is huge
# ZZ Repair data
Nerror_distant_zz =[0,8,47,189,532,843]
Nerror_common_zz =[0,0,28,135,428,812,968]
Nerror_onlyx_zz =[0,2,14,119,432,807,966]
Nerror_onlyz_zz =[0,0,10,97,390,766,961]
# ZX Repair data
Nerror_distant_zx =[0,1,17,88,390,761,976]
Nerror_common_zx =[0,1,12,81,400,754]
Nerror_onlyx_zx =[0,0,16,92,374,724,953]
Nerror_onlyz_zx =[0,1,16,104,385,742,959]
# Calculate pseudo-logical error threshold
lerror_thr = []
lerror_IBM = []
#Using the exact defintion
for error in perror:
    le = 1-(1-error)**12
    lerror_thr.append(le)
    le_IBM  = 12*error
    lerror_IBM.append(le_IBM)
# Calculate Undamaged logical error rates
lerror_undamaged = np.empty((9,))
lerror_undamaged[:] = np.nan 
      
for i in range(len(Nerror_undamaged)):
    e = Nerror_undamaged[i]
    shots = Nshots_undamaged[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_undamaged[i] = le 
# Calculate error rates from one damage repair
lerror_one = np.empty((9,))
lerror_one[:] = np.nan
for i in range(len(Nerror_onerepair)):
    e = Nerror_onerepair[i]
    shots = Nshots_undamaged[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_one[i] = le     
# Calculate error rates from multiple repairs 
#ZZ Repair
#Distant
lerror_distant_zz = np.empty((9,))
lerror_distant_zz[:] = np.nan
for i in range(len(Nerror_distant_zz)):
    e = Nerror_distant_zz[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_distant_zz[i] = le 
#Common
lerror_common_zz = np.empty((9,))
lerror_common_zz[:] = np.nan
for i in range(len(Nerror_common_zz)):
    e = Nerror_common_zz[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_common_zz[i] = le 
#OnlyX
lerror_onlyx_zz = np.empty((9,))
lerror_onlyx_zz[:] = np.nan
for i in range(len(Nerror_onlyx_zz)):
    e = Nerror_onlyx_zz[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_onlyx_zz[i] = le 
#OnlyZ
lerror_onlyz_zz = np.empty((9,))
lerror_onlyz_zz[:] = np.nan
for i in range(len(Nerror_onlyz_zz)):
    e = Nerror_onlyz_zz[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_onlyz_zz[i] = le 
#ZX Repair
#Distant
lerror_distant_zx = np.empty((9,))
lerror_distant_zx[:] = np.nan
for i in range(len(Nerror_distant_zx)):
    e = Nerror_distant_zx[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_distant_zx[i] = le 
#Common
lerror_common_zx = np.empty((9,))
lerror_common_zx[:] = np.nan
for i in range(len(Nerror_common_zx)):
    e = Nerror_common_zx[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_common_zx[i] = le 
#OnlyX
lerror_onlyx_zx = np.empty((9,))
lerror_onlyx_zx[:] = np.nan
for i in range(len(Nerror_onlyx_zx)):
    e = Nerror_onlyx_zx[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_onlyx_zx[i] = le 
#OnlyZ
lerror_onlyz_zx = np.empty((9,))
lerror_onlyz_zx[:] = np.nan
for i in range(len(Nerror_onlyz_zx)):
    e = Nerror_onlyz_zx[i]
    shots = Nshots_undamaged_multiple[i]
    le_f = e/shots
    le = 1-(1-le_f)**(1/12)
    if le == 0:
        continue
    lerror_onlyz_zx[i] = le 


#Plot variables
PLOTZZ = True
PLOTSINGLE = True
PLOTZX = True
plt.plot(perror,lerror_thr,label ='Pseudo-Threshold Line ',linestyle='dashed') #Threshold line
plt.plot(perror,lerror_IBM,label ='Pseudo-Threshold Line-IBM ',linestyle='dashed') #Threshold line-IBM
plt.plot(perror,lerror_undamaged,label ='[[144,12]] ') # True threshold
if PLOTSINGLE:
    plt.plot(perror,lerror_one,label ='[[143,12]]-1 Damage ') # True threshold
# ZZ plots
if PLOTZZ:
    names = ["[[142,12]]-ZZDistant","[[142,12]]-ZZCommon","[[142,12]]-ZZOnlyX","[[142,12]]-ZZOnlyZ"]
    datasets = [lerror_distant_zz,lerror_common_zz,lerror_onlyx_zz,lerror_onlyz_zz]
    for i in range(len(names)):
        data = datasets[i]
        name = names[i]
        plt.plot(perror,data,label =name)
if PLOTZX:
    names = ["[[142,12]]-ZXDistant","[[142,12]]-ZXCommon","[[142,12]]-ZXOnlyX","[[142,12]]-ZXOnlyZ"]
    datasets = [lerror_distant_zx,lerror_common_zx,lerror_onlyx_zx,lerror_onlyz_zx]
    for i in range(len(names)):
        data = datasets[i]
        name = names[i]
        plt.plot(perror,data,label =name)        
plt.xlabel('Input Physical Error Rate')
plt.ylabel('Logical error rate')
plt.legend(fontsize='large')   # Set the font size of the legend
plt.legend(title='Legend')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')  
plt.savefig("SingleErrorRepair.png")
plt.title("Circuit level noise sims plots-Defect Repairs")
plt.show()    
#plt.show() 
print(lerror_undamaged) 


print(Nshots_undamaged)






