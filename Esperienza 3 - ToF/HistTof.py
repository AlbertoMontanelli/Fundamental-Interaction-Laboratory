import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from array import array
from datetime import datetime



cal, dcal, qcal, dqcal, pcov_cal=np.loadtxt('cal_1e2.txt', unpack=True)
v_prop, dv_prop, m_prop, dm_prop, q_prop, dq_prop, pcov_prop=np.loadtxt('vel_prop.txt', unpack=True)


t0_124 = plb.loadtxt('04-24/T1_ToF_124cm.txt',unpack=True)
t1_124 = plb.loadtxt('04-24/T2_ToF_124cm.txt',unpack=True)
t2_124 = plb.loadtxt('04-24/T3_ToF_124cm.txt',unpack=True)

t0_85 = plb.loadtxt('04-30/T1_ToF_85.5cm.txt',unpack=True)
t1_85 = plb.loadtxt('04-30/T2_ToF_85.5cm.txt',unpack=True)
t2_85 = plb.loadtxt('04-30/T3_ToF_85.5cm.txt',unpack=True)

t0_172= plb.loadtxt('05-07/T1_ToF_172.5cm.txt',unpack=True)
t1_172 = plb.loadtxt('05-07/T2_ToF_172.5cm.txt',unpack=True)
t2_172 = plb.loadtxt('05-07/T3_ToF_172.5cm.txt',unpack=True)

t0_109= plb.loadtxt('05-08/T1_ToF_h162cm_d109cm.txt',unpack=True)
t1_109 = plb.loadtxt('05-08/T2_ToF_h162cm_d109cm.txt',unpack=True)
t2_109 = plb.loadtxt('05-08/T3_ToF_h162cm_d109cm.txt',unpack=True)


t0_160= plb.loadtxt('05-09/T1_ToF_h162cm_d160cm.txt',unpack=True)
t1_160 = plb.loadtxt('05-09/T2_ToF_h162cm_d160cm.txt',unpack=True)
t2_160 = plb.loadtxt('05-09/T3_ToF_h162cm_d160cm.txt',unpack=True)

a=np.loadtxt("04-18/medie_1e2_2.5cm.txt", unpack=True)
b=np.loadtxt("04-23/medie_1e2_100cm.txt", unpack=True)
c=np.loadtxt("04-16/medie_1e2_140cm.txt", unpack=True)
d=np.loadtxt("04-17/medie_1e2_220cm.txt", unpack=True)
e=np.loadtxt("04-19/medie_1e2_262.5cm.txt", unpack=True)
f=np.loadtxt("05-02/medie_1e2_180cm.txt", unpack=True)

sigma_arr=np.array([a[2], b[2], c[2], d[2], e[2], f[2]])
sigma=(np.mean(sigma_arr))/2
print(sigma)

h = 108.5
d = np.array([85.5,124.,172.5])

#Funzioni per trovare la posizione in cui è passato il muone e il ToF
def X(t0,t1,q,m):
	return (((t0 -t1)-q)/m)


def Tof1(t0,t1,t2, q):
	return t2-((t0+t1)/2)+ (q/2)

err_Tof1= np.sqrt( (sigma)**2+2*((0.5*sigma)**2) +(0.5*dqcal)**2 )
	
def X_err(t0, t1, m, q, m_err, q_err, cov_mq):
	# Calcolo delle derivate parziali rispetto ai parametri
	dE_dm = -((t0-t1)-q) / (m**2)
	dE_dq = -1 / m
	dE_dt0 = 1 / m
	dE_dt1= -1/m

	# Calcolo dell'errore utilizzando le derivate parziali
	dX = np.sqrt((dE_dm * m_err)**2 + (dE_dq * q_err)**2 + (dE_dt0 * sigma)**2+ (dE_dt1 * sigma)**2 + 2 * dE_dm * dE_dq * cov_mq)
    
	return dX
	
def d_err(x, d, h, sigma_x):
	df_dx = (x-d)/(np.sqrt((x-d)**2+h**2))
	df_dd = (-x+d)/(np.sqrt((x-d)**2+h**2))
	df_dh = (h)/(np.sqrt((x-d)**2+h**2))
	return np.sqrt((df_dx * sigma_x)**2 + (df_dd)**2 + (df_dh)**2)
	
#*******************************************************************************************************************************************************************************

# d=85.5cm h=108.5cm

#*******************************************************************************************************************************************************************************


tof1_85= Tof1(t0_85,t1_85,t2_85, qcal)
x1_85 = X(t0_85,t1_85,qcal,cal)


tof_85new = np.array([])
x_85new =  np.array([])
x_85_err=np.array([])

for i in range(len(x1_85)):
	if ((x1_85[i]<0.) | (x1_85[i]>280.)):
		pippo = 1
	else:
		tof_85new = np.append(tof_85new, tof1_85[i])
		x_85new = np.append(x_85new, x1_85[i])
		x_85_err=np.append(x_85_err, X_err(t0_85[i], t1_85[i], cal, qcal, dcal, dqcal, pcov_cal))

d_85 = np.sqrt((x_85new-85.5)**2 + 108.5**2)
d_85_err = d_err(x_85new, 85.5, 108.5, x_85_err)

#*******************************************************************************************************************************************************************************



#*****************************************************************************************************************************************************************************

# d=109cm h=162cm

#*******************************************************************************************************************************************************************************

tof1_109= Tof1(t0_109,t1_109,t2_109, qcal)
x1_109 = X(t0_109,t1_109,qcal,cal)

tof_109new = np.array([])
x_109new =  np.array([])
x_109_err=np.array([])

for i in range(len(x1_109)):
	if ((x1_109[i]<0.) | (x1_109[i]>280.)):
		pippo = 1
	else:
		tof_109new = np.append(tof_109new, tof1_109[i])
		x_109new = np.append(x_109new, x1_109[i])
		x_109_err=np.append(x_109_err, X_err(t0_109[i], t1_109[i], cal, qcal, dcal, dqcal, pcov_cal))

d_109 = np.sqrt((x_109new-109.)**2 + 162.**2)
d_109_err=d_err(x_109new, 109., 162., x_109_err)


#******************************************************************************************************************************************************************************************


#*******************************************************************************************************************************************************************************

# d=124cm h=108.5cm

#*******************************************************************************************************************************************************************************

tof1_124= Tof1(t0_124,t1_124,t2_124, qcal)
x1_124 = X(t0_124,t1_124,qcal,cal)

tof_124new = np.array([])
x_124new =  np.array([])
x_124_err=np.array([])

for i in range(len(x1_124)):
	if ((x1_124[i]<0.) | (x1_124[i]>280.)):
		pippo = 1
	else:
		tof_124new = np.append(tof_124new, tof1_124[i])
		x_124new = np.append(x_124new, x1_124[i])
		x_124_err=np.append(x_124_err, X_err(t0_124[i], t1_124[i], cal, qcal, dcal, dqcal, pcov_cal))

d_124 = np.sqrt((x_124new-124.)**2 + 108.5**2)
d_124_err=d_err(x_124new, 124., 108.5, x_124_err)

#*******************************************************************************************************************************************************************************


#*******************************************************************************************************************************************************************************

# d=172.5cm h=108.5cm

#*******************************************************************************************************************************************************************************

tof1_172= Tof1(t0_172,t1_172,t2_172, qcal)
x1_172 = X(t0_172,t1_172,qcal,cal)

tof_172new = np.array([])
x_172new =  np.array([])
x_172_err=np.array([])

for i in range(len(x1_172)):
	if ((x1_172[i]<0.) | (x1_172[i]>280.)):
		pippo = 1
	else:
		tof_172new = np.append(tof_172new, tof1_172[i])
		x_172new = np.append(x_172new, x1_172[i])
		x_172_err=np.append(x_172_err, X_err(t0_172[i], t1_172[i], cal, qcal, dcal, dqcal, pcov_cal))

d_172 = np.sqrt((x_172new-172.5)**2 + 108.5**2)
d_172_err=d_err(x_172new, 172.5, 108.5, x_172_err)

#*******************************************************************************************************************************************************************************



#*******************************************************************************************************************************************************************************

# d=160cm h=162cm

#*******************************************************************************************************************************************************************************

tof1_160= Tof1(t0_160,t1_160,t2_160, qcal)
x1_160 = X(t0_160,t1_160,qcal,cal)

tof_160new = np.array([])
x_160new =  np.array([])
x_160_err=np.array([])

for i in range(len(x1_160)):
	if ((x1_160[i]<0.) | (x1_160[i]>280.)):
		pippo = 1
	else:
		tof_160new = np.append(tof_160new, tof1_160[i])
		x_160new = np.append(x_160new, x1_160[i])
		x_160_err=np.append(x_160_err, X_err(t0_160[i], t1_160[i], cal, qcal, dcal, dqcal, pcov_cal))

d_160 = np.sqrt((x_160new-160.)**2 + 162.**2)
d_160_err=d_err(x_160new, 160., 162., x_160_err)

#*******************************************************************************************************************************************************************************

diff_124=t0_124-t1_124
sum_124=t0_124+t1_124



c1=TCanvas("hist diff ToF d=124cm", "hist diff ToF d=124cm") 
num_bin = 300

# Crea un istogramma con il numero di bin specificato
hist1=TH1F("hist diff ToF d=124cm", "hist diff ToF d=124cm", num_bin, -30, 20)
for i in range(len(diff_124)):
	hist1.Fill(diff_124[i])


hist1.Draw()
c1.Draw()

c2=TCanvas("hist sum ToF d=124cm", "hist sum ToF d=124cm") 
num_bin = 200

# Crea un istogramma con il numero di bin specificato
hist2=TH1F("hist sum ToF d=124cm", "hist sum ToF d=124cm", num_bin, 220, 280)
for i in range(len(sum_124)):
	hist2.Fill(sum_124[i])


hist2.Draw()
c2.Draw()

input()

