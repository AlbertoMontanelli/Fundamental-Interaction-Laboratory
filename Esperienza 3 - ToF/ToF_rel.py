import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
#from ROOT import *
from array import array
from datetime import datetime
import sympy as sp


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


dist = np.concatenate((d_85, d_109, d_160, d_124, d_172))
dist_err=np.concatenate((d_85_err, d_109_err, d_160_err, d_124_err, d_172_err))
print('dist:')
print(dist_err)
print(np.sum(dist_err**2))
print(np.sqrt(np.sum(dist_err**2))/len(dist_err))
#err_d_medie = np.concatenate((err_d_85_medie, err_d_109_medie, err_d_160_medie, err_d_124_medie, err_d_172_medie))
tof = np.concatenate((tof_85new, tof_109new, tof_160new, tof_124new, tof_172new))
tof_err= np.array([sigma])*np.ones(len(tof))
print('\n')
print('tof:')
print(tof_err)
print(np.sum(tof_err**2))
print(np.sqrt(np.sum(tof_err**2))/len(tof_err))
#err_tof_medie = np.concatenate((err_tof_85_medie, err_tof_109_medie, err_tof_160_medie, err_tof_124_medie, err_tof_172_medie))


if( (len(dist))==(len(dist_err))==(len(tof))==(len(tof_err))):
	print(len(dist))

plt.figure(' Plot')
plt.title('Scatter plot eventi di passaggio raggi cosmici' , fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })
plt.plot(dist, tof, '.')
plt.grid()
plt.xlabel('distanza percorsa [cm]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })
plt.ylabel('ToF [ns]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })
plt.text(0.95, 0.05, '41821 eventi\nAcquisizioni dal 24/04/24 al 09/05/24\n h PMT03: (108, 162.5)cm\n d PMT03: (85.5, 109, 124, 160, 172.5)cm', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 34, }, horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi

plt.savefig('grafici/scatter_plot.pdf', dpi=1200, bbox_inches='tight')



plt.show()

d_s = dist

d_s= sorted(d_s)

d = np.array([])

for i in range(0, len(dist), 5226): #3216 per 13 punti
	d=np.append(d, d_s[i])




#intervalli_distanza= [ (d[0], d[1]),(d[1], d[2]), (d[2], d[3]), (d[3],d[4]), (d[4], d[5]), (d[5], d[6]), (d[6], d[7]), (d[7], d[8]), (d[8], d[9]), (d[9], d[10]), (d[10], d[11]), (d[11], d[12]) ]
intervalli_distanza= [ (d[0], d[1]),(d[1], d[2]), (d[2], d[3]), (d[3],d[4]), (d[4], d[5]), (d[5], d[6]), (d[6], d[7])]

d_medie = np.array([])
err_d_medie = np.array([])
tof_medie = np.array([])
err_tof_medie = np.array([])
tempo_err=np.array([])
distanza_err=np.array([])

# Calcola la distanza e il tempo medio per ogni intervallo
for intervallo in intervalli_distanza:
	
	distanza_min, distanza_max = intervallo
	tempo_intervallo = tof[(dist >= distanza_min) & (dist < distanza_max)]
	distanza_intervallo = dist[(dist>= distanza_min) & (dist < distanza_max)]
	tempo_intervallo_err = tof_err[(dist >= distanza_min) & (dist < distanza_max)]
	distanza_intervallo_err = dist_err[(dist>= distanza_min) & (dist < distanza_max)]
	
	distanza_media = np.mean(distanza_intervallo)
	tempo_medio = np.mean(tempo_intervallo)
    
	d_medie = np.append(d_medie, distanza_media)
	tof_medie = np.append(tof_medie, tempo_medio)
    
	err_d_medie = np.append(err_d_medie, np.sqrt(np.sum(distanza_intervallo_err**2))/len(distanza_intervallo_err) )
	err_tof_medie = np.append(err_tof_medie, np.sqrt(np.sum(tempo_intervallo_err**2))/len(tempo_intervallo_err))

	tempo_err=np.append(tempo_err, np.sqrt(np.sum((tempo_intervallo-tempo_medio)**2)/len(tempo_intervallo))/np.sqrt(len(tempo_intervallo)))
	distanza_err=np.append(distanza_err, np.sqrt(np.sum((distanza_intervallo-distanza_media)**2)/len(distanza_intervallo))/np.sqrt(len(distanza_intervallo)))


print(tempo_err)
print(err_tof_medie)

print('\n')

print(distanza_err)
print(err_d_medie)

x = np.linspace(min(d_medie), max(d_medie), 1000)



def retta(x, m, q):
    return m * x + q
##VELOCITA DI PROPAGAZIONE 

# Eseguire il fit dei dati
popt, pcov = curve_fit(retta, d_medie, tof_medie, sigma=err_tof_medie, absolute_sigma=False)

# Estraiamo i parametri del fit e i relativi errori
m, q = popt
m_err, q_err = np.sqrt(np.diag(pcov))

# Calcolo del chi-quadro
residuals = tof_medie - retta(d_medie, *popt)
#print(residuals.mean())
chi_squared = np.sum((residuals / err_tof_medie) ** 2)
degrees_of_freedom = len(d_medie) - len(popt)

plt.figure('Retta Def')
plt.errorbar(d_medie, tof_medie, err_tof_medie, err_d_medie, 'o', color = 'red', linestyle = ' ', ecolor='red', elinewidth=0.8, markersize=1.5, capsize=2.5, capthick=0.8)
plt.plot(x, retta(x, m, q))
plt.grid()
plt.xlabel('distanza percorsa [cm]')
plt.ylabel('ToF [ns]')


legenda = "Parametri del fit:\n"
legenda += "$m$ = {:.4f} $\pm$ {:.4f}\n".format(m, m_err)
legenda += "$q$ = {:.4f} $\pm$ {:.4f}\n".format(q, q_err)
legenda += "$v = $ = {:.4f} $\pm$ {:.4f}\n".format((1/m), (m_err*(1/(m**2))))
legenda += "$\chi^2$ / ndof = {:.2f} / {:d}".format(chi_squared,degrees_of_freedom)
plt.text(0.1, 0.9, legenda, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))




plt.show()




