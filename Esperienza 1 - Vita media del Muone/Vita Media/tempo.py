import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os


#VITA MEDIA
ch, t= plb.loadtxt("MV_acquisizione6-12.dat",unpack=True)

t1=np.array([])
t2=np.array([])

for i,element in enumerate(ch):
	if(element==2):
		t2=np.append(t2,t[i])
	if(element==1):
		t1=np.append(t1,t[i])



temp = 0.

T = np.array([])
for i in range(len(t2)):
	for j in range(len(t1)):
		temp = (t2[i]-t1[j])
		if 0 <= temp <= 10e-6:
			T = np.append(T,temp)
			
a=plb.savetxt("MV_acquisizione6-12.txt", T)
			

			
