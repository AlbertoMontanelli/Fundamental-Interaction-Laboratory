import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from array import array
from datetime import datetime

#Funzioni per trovare la posizione in cui è passato il muone e il ToF
def X(t0,t1,q,m):
	return (((t0 -t1)-qcal)/cal)


def Tof1(t0,t1,t2, q):
	return t2-((t0+t1)/2) + (q/2)
	
	
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

cal, dcal, qcal, dqcal, pcov_cal=np.loadtxt('cal_1e2.txt', unpack=True)
v_prop, dv_prop, m_prop, dm_prop, q_prop, dq_prop, pcov_prop=np.loadtxt('vel_prop.txt', unpack=True)

sigma=0.3837003618317166
err_Tof1= np.sqrt( (sigma)**2+2*((0.5*sigma)**2) +(0.5*dqcal)**2 )


a1 = plb.loadtxt('05-17/T1_ToF_decay.txt',unpack=True)
a2 = plb.loadtxt('05-17/T2_ToF_decay.txt',unpack=True)
a3 = plb.loadtxt('05-17/T3_ToF_decay.txt',unpack=True)

b1 = plb.loadtxt('05-21/T1_ToF_decay.txt',unpack=True)
b2 = plb.loadtxt('05-21/T2_ToF_decay.txt',unpack=True)
b3 = plb.loadtxt('05-21/T3_ToF_decay.txt',unpack=True)

c1 = plb.loadtxt('05-22/T1_ToF_decay.txt',unpack=True)
c2 = plb.loadtxt('05-22/T2_ToF_decay.txt',unpack=True)
c3 = plb.loadtxt('05-22/T3_ToF_decay.txt',unpack=True)

d1 = plb.loadtxt('05-22/T1_ToF_decay_2.txt',unpack=True)
d2 = plb.loadtxt('05-22/T2_ToF_decay_2.txt',unpack=True)
d3 = plb.loadtxt('05-22/T3_ToF_decay_2.txt',unpack=True)

e1 = plb.loadtxt('05-23/T1_ToF_decay.txt',unpack=True)
e2 = plb.loadtxt('05-23/T2_ToF_decay.txt',unpack=True)
e3 = plb.loadtxt('05-23/T3_ToF_decay.txt',unpack=True)

f1 = plb.loadtxt('05-23/T1_ToF_decay_2.txt',unpack=True)
f2 = plb.loadtxt('05-23/T2_ToF_decay_2.txt',unpack=True)
f3 = plb.loadtxt('05-23/T3_ToF_decay_2.txt',unpack=True)

print(len(a1)+len(b1)+len(c1)+len(d1)+len(e1)+len(f1))

t_pmt1=np.concatenate((c1, d1, e1, f1))
t_pmt2=np.concatenate((c2, d2, e2, f2))
t_pmt3=np.concatenate((c3, d3, e3, f3))


tof_nonrel= Tof1(t_pmt1,t_pmt2,t_pmt3, qcal)
x_nonrel = X(t_pmt1,t_pmt2,qcal,cal)
tof_nonrelnew = np.array([])
x_nonrelnew =  np.array([])
x_err=np.array([])

tof_nonrel_17may= Tof1(a1, a2, a3, qcal)
x_nonrel_17may = X(a1,a2, qcal,cal)
tof_nonrelnew_17may = np.array([])
x_nonrelnew_17may =  np.array([])
x_err_17may=np.array([])

pippo = 0


for i in range(len(x_nonrel)):
	if ((x_nonrel[i]<0) | (x_nonrel[i]>280)):
		pippo = 1
	else:
		tof_nonrelnew = np.append(tof_nonrelnew, tof_nonrel[i])
		x_nonrelnew = np.append(x_nonrelnew, x_nonrel[i])
		x_err=np.append(x_err, X_err(t_pmt1[i], t_pmt2[i], cal, qcal, dcal, dqcal, pcov_cal))
		
for i in range(len(x_nonrel_17may)):
	if ((x_nonrel_17may[i]<0) | (x_nonrel_17may[i]>280)):
		pippo = 1
	else:
		tof_nonrelnew_17may = np.append(tof_nonrelnew_17may, tof_nonrel_17may[i])
		x_nonrelnew_17may = np.append(x_nonrelnew_17may, x_nonrel_17may[i])
		x_err_17may=np.append(x_err_17may, X_err(a1[i], a2[i], cal, qcal, dcal, dqcal, pcov_cal))

d_nonrel = np.sqrt((x_nonrelnew-150)**2 + 166**2)
dist_err1=d_err(x_nonrelnew, 150., 166., x_err)

d_nonrel_17may = np.sqrt((x_nonrelnew_17may-148)**2 + 170**2)
dist_err_17may=d_err(x_nonrelnew_17may, 148., 170., x_err_17may)

dist_err=np.concatenate((dist_err1, dist_err_17may))

'''
plt.figure('non rel ')
plt.title('non rel' )
plt.plot(d_nonrel, tof_nonrelnew, '.')
plt.grid()
plt.xlabel('distanza percorsa [cm]')
plt.ylabel('ToF [ns]')
plt.show()
'''

v_nonrel = d_nonrel /(tof_nonrelnew + 28.011610204385715)
v_nonrel_17may = d_nonrel_17may /(tof_nonrelnew_17may + 28.011610204385715)

v=np.concatenate((v_nonrel, v_nonrel_17may))



unosubeta=29.9792458/v_nonrel

np.savetxt('v_decay_alberto.txt', unosubeta)
std=np.std(unosubeta)/(np.sqrt(len(unosubeta)))
media=np.mean(unosubeta)
print('media di 1/beta', media)
print('std di 1/beta', std)

print('beta:', 1/media)
print('std:', std*((1/media)**2))


"""
mediana=np.median(beta_mt)
tenperc=int(0.05*len(beta_mt))
beta_mt=beta_mt[:-tenperc]
media=np.mean(beta_mt)
std=np.std(beta_mt)/np.sqrt(len(beta_mt))
print(media, std)
"""
c1=TCanvas("Non rel", "Non rel")

num_bin = 250

# Crea un istogramma con il numero di bin specificato
hist1=TH1F("V_nonrel", "1/#beta decadimenti; 1/#beta; eventi", num_bin, -1.5, 3)
for i in range(len(unosubeta)):
	hist1.Fill(unosubeta[i])

hist1.Draw()
w_mean=hist1.GetMean()
w_std=hist1.GetRMS()




weighted_sum = 0.0
total_weight = 0.0
    
# Itera sui bin dell'istogramma
for i in range(0, hist1.GetNbinsX()+1 ):
	bin_center = hist1.GetBinCenter(i)
	bin_content = hist1.GetBinContent(i)
        
	weighted_sum += bin_center * bin_content
	total_weight += bin_content
    
	# Calcola la media pesata
	weighted_mean = weighted_sum / total_weight if total_weight != 0 else 0
    
# Stampa il risultato
print(f"Media Pesata: {1/weighted_mean}")


#timestamp
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
timestamp_text = "Timestamp: " + current_time
latex = TLatex() # Creazione di un TLatex
latex.SetTextSize(0.03)
latex.SetNDC(True)  # Imposta le coordinate normalizzate del canvas
latex.DrawLatex(0.7, 0.05, timestamp_text) # Imposta la posizione del testo sul canvas (in questo caso nell'angolo in basso a destra)


legend3 = TLegend(0.5, 0.5, 0.9, 0.8)
# Aggiungi il numero di entries dell'istogramma alla legenda

legend3.AddEntry(hist1, f"Histogram Entries: {hist1.GetEntries():.0f}", "l")
legend3.AddEntry("", f"bin: {num_bin}", "")
#legend3.AddEntry("", f"Posizione : d = 148 cm , \Delta h = 170 cm ", "")
legend3.AddEntry("", f"1/#beta = {(media):.2f} #pm{(std):.2f} ", "")
legend3.AddEntry("", f"#beta = {(1/media):.2f} #pm{(std*((1/media)**2)):.2f} ", "")





legend3.Draw()

c1.Draw()

input()


"""
#*******************************************************************************************************************************************************************************************

# SCATTER PLOT

dist=np.concatenate((d_nonrel, d_nonrel_17may))
tof=np.concatenate((tof_nonrelnew, tof_nonrelnew_17may))

plt.figure(' Plot')
plt.title('Mega Scatter Plot' )
plt.plot(dist, tof, '.')
plt.grid()
plt.xlabel('distanza percorsa [cm]')
plt.ylabel('ToF [ns]')
plt.show()

#*********************************************************************************************************************************************************************************************





#*********************************************************************************************************************************************************************************************

# VELOCITÀ MEDIA DA SCATTER PLOT


tof_err= np.array([sigma])*np.ones(len(tof))
	

	


d_s = dist

d_s= sorted(d_s)

d = np.array([])

for i in range(0, len(dist), 243): #3216 per 13 punti
	d=np.append(d, d_s[i])




#intervalli_distanza= [ (d[0], d[1]),(d[1], d[2]), (d[2], d[3]), (d[3],d[4]), (d[4], d[5]), (d[5], d[6]), (d[6], d[7]), (d[7], d[8]), (d[8], d[9]), (d[9], d[10]), (d[10], d[11]), (d[11], d[12]) ]
intervalli_distanza= [ (d[0], d[1]),(d[1], d[2]), (d[2], d[3]), (d[3],d[4]), (d[4], d[5])]

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

#*********************************************************************************************************************************************************************************************
"""



