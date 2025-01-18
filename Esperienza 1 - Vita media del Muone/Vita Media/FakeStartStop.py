import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
#from ROOT import *


ch, t= plb.loadtxt('MV_acquisizione6-12.dat',unpack=True)

dt1=np.array([])
dt2=np.array([])
for i in range(1, len(ch)-1, 1):
	if ( (ch[i]==1) & (ch[i-1]==1) & (0<=t[i]-t[i-1]<=20e-6)):
		dt1=np.append(dt1, t[i]-t[i-1])
	if ( (ch[i]==2) & (ch[i-1]==2) & (0<=t[i]-t[i-1]<=20e-6)):
		dt2=np.append(dt2, t[i]-t[i-1])


	
np.savetxt('FakeStart.txt',dt1)	
np.savetxt('FakeStop.txt',dt2)	

print(len(dt1)+len(dt2))


