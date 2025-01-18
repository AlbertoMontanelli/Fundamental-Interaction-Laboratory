import pylab
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import time
from ROOT import *


ch1, t1= pylab.loadtxt('11-28.dat',unpack=True)
ch2, t2= pylab.loadtxt('11-29.dat',unpack=True)
ch3, t3= pylab.loadtxt('11-30_newsetup.dat',unpack=True)
ch4, t4= pylab.loadtxt('12-05_newsetup.dat',unpack=True)
ch5, t5= pylab.loadtxt('12-06_newsetup.dat',unpack=True)
ch6, t6= pylab.loadtxt('12-12_newsetup.dat',unpack=True)

ch=np.concatenate((ch1, ch2))
t=np.concatenate((t1, t2))

ch_newsetup=np.concatenate((ch3, ch4, ch5, ch6))
t_newsetup=np.concatenate((t3, t4, t5, t6))


i=len(ch)-1

while i>0:
	
	if ( (ch[i]==1) & (ch[i-1]==1) & (0<=t[i]-t[i-1]<=20e-6)):
		ch=np.delete(ch,i)
		ch=np.delete(ch,i-1)
		t=np.delete(t,i)
		t=np.delete(t,i-1)

	if ( (ch[i]==2) & (ch[i-1]==2) & (0<=t[i]-t[i-1]<=20e-6)):
		ch=np.delete(ch,i)
		ch=np.delete(ch,i-1)
		t=np.delete(t,i)
		t=np.delete(t,i-1)

	i=i-2

dt = np.array([])
dt_newsetup=np.array([])
	

for i in range (len(ch)-1):
	if((ch[i]==1) & (ch[i+1]==2) & (200e-9<=t[i+1]-t[i]<=20e-6)):
		dt =np.append(dt,t[i+1]-t[i])

for i in range (len(ch_newsetup)-1):
	if((ch_newsetup[i]==1) & (ch_newsetup[i+1]==2) & (200e-9<=t_newsetup[i+1]-t_newsetup[i]<=20e-6)):
		dt_newsetup =np.append(dt_newsetup,t_newsetup[i+1]-t_newsetup[i])
		
np.savetxt('vita_media.txt',dt)
np.savetxt('vita_media_newsetup.txt', dt_newsetup)










