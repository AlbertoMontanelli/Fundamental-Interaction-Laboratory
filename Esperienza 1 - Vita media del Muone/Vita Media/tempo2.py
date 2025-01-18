import pylab
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import time
#from ROOT import *



ch, t= pylab.loadtxt('MV_acquisizione6-12.dat',unpack=True)


j=0
k=0
i=1

"""
while i<len(ch):
	if ( (ch[i]==1) & (ch[i-1]==1) & (0<=t[i]-t[i-1]<=25e-6)):
		ch=np.delete(ch,i)
		t=np.delete(t,i)
		j=j+1
	if ( (ch[i]==2) & (ch[i-1]==2) & (0<=t[i]-t[i-1]<=25e-6)):
		ch=np.delete(ch,i)
		t=np.delete(t,i)
		k=k+1
	i=i+1

"""

dt = np.array([])

for i in range (0, len(ch)-1, 1):
	if ((i<=len(ch)-2) & (ch[i]==2) & (ch[i+1]==1)):
		j=i-1	
		while((ch[j]==1) & (0<= (t[i]-t[j])<= 20e-6)):
			dt = np.append(dt, t[i]-t[j])
			j=j-1
	if ((i<=len(ch)-2) & (ch[i]==2) & (ch[i+1]==2)):
		j=i-1
		while((ch[j]==1) & (0<= (t[i+1]-t[j])<= 20e-6)):
			dt = np.append(dt, t[i+1]-t[j])
			j=j-1
	if ( (i==len(ch)-1) & (ch[i]==2) ):
		j=i-1	
		while((ch[j]==1) & (0<= (t[i]-t[j])<= 20e-6)):
			dt = np.append(dt, t[i]-t[j])
			j=j-1
			
np.savetxt('MV_acquisizione6-12-20us-unclean.txt',dt)

print(len(dt))

