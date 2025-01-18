import pylab
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import time
#from ROOT import *

start = time.time()

ch, t= pylab.loadtxt('MV_acquisizione12-12.dat',unpack=True)

a=len(ch)


j=0
k=0
i=1
while i<len(ch):
	if ( (ch[i]==1) & (ch[i-1]==1) & (0<=t[i]-t[i-1]<=20e-6)):
		ch=np.delete(ch,i)
		ch=np.delete(ch,i-1)
		t=np.delete(t,i)
		t=np.delete(t,i-1)
		j=j+2
	if ( (ch[i]==2) & (ch[i-1]==2) & (0<=t[i]-t[i-1]<=20e-6)):
		ch=np.delete(ch,i)
		ch=np.delete(ch,i-1)
		t=np.delete(t,i)
		t=np.delete(t,i-1)
		k=k+2
	i=i+1
	
	



dt = np.array([])

	

for i in range (1, len(ch)-1, 1):
	if((ch[i]==2) & (0<=t[i]-t[i-1]<=20e-6)):
		dt =np.append(dt,t[i]-t[i-1])


np.savetxt('MV_acquisizione12-12-20us-clean.txt',dt)

tem = time.time()-start

#print(tem)
print(len(dt))









