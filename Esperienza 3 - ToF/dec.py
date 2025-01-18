import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d


#Funzioni per trovare la posizione in cui è passato il muone e il ToF
def X(t0,t1,q,m):
	return (((t0 -t1)-qcal)/cal)


def Tof1(t0,t1,t2, q):
	return t2-((t0+t1)/2) + (q/2)


cal, dcal, qcal, dqcal, pcov_cal=np.loadtxt('cal_1e2.txt', unpack=True)
v_prop, dv_prop, m_prop, dm_prop, q_prop, dq_prop, pcov_prop=np.loadtxt('vel_prop.txt', unpack=True)



#fino al 17-05
D = 148 
h = 170
#dal 21/05
d = 150 
h = 166 



ch, t = np.loadtxt('05-22/Decay_05-22_2.dat', usecols=(0, 1), unpack=True)


print('canali tot', len(ch))

a = 0
b = 0
c = 0
# Itera su ogni evento nel tuo albero

for i in range(len(ch)-1):
	if(ch[i]==1):
		b = b+1

	if((ch[i]==1) and (ch[i+1] == 2)) :
		a = a+1

print('a', a)
print('b', b)

	
    
#print(ch[35])


# Apri il file di root contenente il tuo albero
file = TFile.Open("05-22/Decay_05-22_DRS2.root")


# Ottieni l'albero dal file
tree = file.Get("rec")

# numero di eventi che ci sono
N=tree.GetEntries()

print('entry', N)

#tree.Scan('t0:t1:t2')

# Inizializza gli array bidimensionali per w1 e t1, composti da N righe e 1024 colonne corrispondenti alle N righe
w0 = np.zeros((N, 1024))
t0 = np.zeros((N, 1024))
w1 = np.zeros((N, 1024))
t1 = np.zeros((N, 1024))
w2 = np.zeros((N, 1024))
t2 = np.zeros((N, 1024))
a = 0
b = 0
c = 0
# Itera su ogni evento nel tuo albero

for i in range(len(ch)-1):
	if(ch[i]==1):
		b = b+1

	if((ch[i] ==1) and (ch[i+1] == 2)) :
		a = a+1
		#print(i)

		c = i+1-a
		
	

		# Ottieni l'evento i-esimo
		tree.GetEntry(c)

		#print(i)
	
		# Ottieni i valori dei rami w1 e t1 per l'evento i-esimo
		w0_event = tree.w0
		t0_event = tree.t0
		w1_event = tree.w1
		t1_event = tree.t1
		w2_event = tree.w2
		t2_event = tree.t2
		
		#print(c,'ok')
		# Riempi l'array bidimensionale con i valori dell'evento i-esimo
		for j in range(1024):
			w0[c][j] = w0_event[j]
			t0[c][j] = t0_event[j]
			w1[c][j] = w1_event[j]
			t1[c][j] = t1_event[j]
			w2[c][j] = w2_event[j]
			t2[c][j] = t2_event[j]


print('ch2', a)
print('ch1', b)


# Inizializza gli array per salvare i minimi e i tempi corrispondenti
min_t0 = np.array([])
min_t1 = np.array([])
min_t2 = np.array([])

dT_slope_0e1=np.array([])
dT_slope_1e2=np.array([])
dT_slope_0e2=np.array([])
t_pmt1=np.array([])
t_pmt2=np.array([])
t_pmt3=np.array([])



for i in range(N):

	max_w0=np.max(w0[i])
	min_w0 =np.min(w0[i])

	
	# Trova l'indice corrispondente del minimo valore di w0 per l'evento i
	min_indices_w0 = int(np.argmin(w0[i]))
	t0_soglia=np.array([])
	
	
	max_w1=np.max(w1[i])
	min_w1 = np.min(w1[i])
	min_indices_w1 = int(np.argmin(w1[i]))
	t1_soglia=np.array([])

	
	
	max_w2=np.max(w2[i])
	min_w2 = np.min(w2[i])
	min_indices_w2 = int(np.argmin(w2[i]))
	t2_soglia=np.array([])


	
	
	
	if( (t0[i][min_indices_w0]>10) & (t1[i][min_indices_w1]>10) & (t2[i][min_indices_w2]>10) & (t2[i][min_indices_w2]<t0[i][min_indices_w0]) & (t2[i][min_indices_w2]<t1[i][min_indices_w1]) & (max_w0<=20e-3) & (min_w0>=-500e-3) & (max_w1<=20e-3) & (min_w1>=-500e-3) & (max_w2<=20e-3) & (min_w2>=-500e-3)):
		
#TEMPO soglia T0 (PMT01)
		j=0		
		while(t0[i][j]<=5):
			j=j+1
		
		while(w0[i][j]>=0.40*min_w0):
			j=j+1	


		j0=j
		t0_control=(t0[i][min_indices_w0]-t0[i][j])/2
		b0=int(j)
		
		
		while (t0[i][b0] <= (t0[i][j] + t0_control)):
			if ( ( (w0[i][b0]>=0.40*min_w0) & (w0[i][b0+1]<0.40*min_w0)) | ( (w0[i][b0]<0.40*min_w0) & (w0[i][b0+1]>=0.40*min_w0)) ):
				t0_soglia=np.append(t0_soglia, t0[i][b0])
				b0=b0+1
			else:
				b0=b0+1


#TEMPO soglia T1 (PMT02)
		j=0
		while(t1[i][j]<=5):
			j=j+1
		
		while(w1[i][j]>=0.40*min_w1):
			j=j+1	

		j1=j
		t1_control=(t1[i][min_indices_w1]-t1[i][j])/2
		b1=int(j)
		
		while (t1[i][b1] <= (t1[i][j] + t1_control)):
			if ( ( (w1[i][b1]>=0.40*min_w1) & (w1[i][b1+1]<0.40*min_w1)) | ( (w1[i][b1]<0.40*min_w1) & (w1[i][b1+1]>=0.40*min_w1)) ):
				t1_soglia=np.append(t1_soglia, t1[i][b1])
				b1=b1+1
			else:
				b1=b1+1
					
		
		
#Tempo soglia T2 (PMT03)
		j=0
		while(t2[i][j]<=5):
			j=j+1
		
		while(w2[i][j]>=0.40*min_w2):
			j=j+1	
		

		j2=j
		t2_control=(t2[i][min_indices_w2]-t2[i][j])/2
		b2=int(j)
		
		while (t2[i][b2] <= (t2[i][j] + t2_control)):
			if ( ( (w2[i][b2]>=0.40*min_w2) & (w2[i][b2+1]<0.40*min_w2)) | ( (w2[i][b2]<0.40*min_w2) & (w2[i][b2+1]>=0.40*min_w2)) ):
				t2_soglia=np.append(t2_soglia, t2[i][b2])
				b2=b2+1
			else:
				b2=b2+1
					
		


		if ( ((len(t0_soglia))>=1) | ((len(t1_soglia))>=1) | ((len(t2_soglia))>=1)):
			pippo=1
			
		else:
		
			# Definisci i tuoi quattro punti come coordinate x e y
			x0 = np.array([t0[i][j0-2], t0[i][j0-1], t0[i][j0+1], t0[i][j0+2]])  
			y0 = np.array([w0[i][j0-2], w0[i][j0-1], w0[i][j0+1], w0[i][j0+2]])

			# Crea una funzione di interpolazione polinomiale
			f0= interp1d(x0, y0, kind='cubic')
			x0_new = np.linspace(t0[i][j0-2], t0[i][j0+2], 10000)
			y0_linear = f0(x0_new)

			for k0 in range(len(x0_new)):
				if (y0_linear[k0]<=0.40*min_w0):
					t0_soglia=None
					t0_soglia=x0_new[k0]
					break	
		
			# Definisci i tuoi quattro punti come coordinate x e y
			x1 = np.array([t1[i][j1-2], t1[i][j1-1], t1[i][j1+1], t1[i][j1+2]])  
			y1 = np.array([w1[i][j1-2], w1[i][j1-1], w1[i][j1+1], w1[i][j1+2]])

			# Crea una funzione di interpolazione polinomiale
			f1 = interp1d(x1, y1, kind='cubic')
			x1_new = np.linspace(t1[i][j1-2], t1[i][j1+2], 10000)
			y1_linear = f1(x1_new)
			for k1 in range(len(x1_new)):
				if (y1_linear[k1]<=0.40*min_w1):
					t1_soglia=None
					t1_soglia=x1_new[k1]
					break


			# Definisci i tuoi quattro punti come coordinate x e y
			x2 = np.array([t2[i][j2-2], t2[i][j2-1], t2[i][j2+1], t2[i][j2+2]])  
			y2= np.array([w2[i][j2-2], w2[i][j2-1], w2[i][j2+1], w2[i][j2+2]])

			# Crea una funzione di interpolazione polinomiale
			f2 = interp1d(x2, y2, kind='cubic')
			x2_new = np.linspace(t2[i][j2-2], t2[i][j2+2], 10000)
			y2_linear = f2(x2_new)
			for k2 in range(len(x2_new)):
				if (y2_linear[k2]<=0.40*min_w2):
					t2_soglia=None
					t2_soglia=x2_new[k2]
					break	
					
			
			dT_slope_0e1=np.append(dT_slope_0e1, t0_soglia-t1_soglia)
			#dT_slope_1e2=np.append(dT_slope_1e2, t1_soglia-t2_soglia)
			#dT_slope_0e2=np.append(dT_slope_0e2, t0_soglia-t2_soglia)

			
			t_pmt1=np.append(t_pmt1, t0_soglia)
			t_pmt2=np.append(t_pmt2, t1_soglia)
			t_pmt3=np.append(t_pmt3, t2_soglia)



print(len(t_pmt1))

"""
np.savetxt('05-23/t0_tagliati.txt', t_pmt1)
np.savetxt('05-23/t1_tagliati.txt', t_pmt2)
np.savetxt('05-23/t2_tagliati.txt', t_pmt3)


t1_1 = plb.loadtxt('05-22/t0_tagliati.txt',unpack=True)
t2_1 = plb.loadtxt('05-22/t1_tagliati.txt',unpack=True)
t3_1 = plb.loadtxt('05-22/t2_tagliati.txt',unpack=True)

t1_2 = plb.loadtxt('05-23/t0_tagliati.txt',unpack=True)
t2_2 = plb.loadtxt('05-23/t1_tagliati.txt',unpack=True)
t3_2 = plb.loadtxt('05-23/t2_tagliati.txt',unpack=True)

t_pmt1 = np.concatenate((t1_1, t1_2))
t_pmt2 = np.concatenate((t2_1, t2_2))
t_pmt3 = np.concatenate((t3_1, t3_2))
"""
tof_nonrel= Tof1(t_pmt1,t_pmt2,t_pmt3, qcal)
x_nonrel = X(t_pmt1,t_pmt2,qcal,cal)

#print('len',len(tof_nonrel))

pippo = 0

tof_nonrelnew = np.array([])
x_nonrelnew =  np.array([])
for i in range(len(x_nonrel)):
	if ((x_nonrel[i]<0) | (x_nonrel[i]>280)):
		pippo = 1
	else:
		tof_nonrelnew = np.append(tof_nonrelnew, tof_nonrel[i])
		x_nonrelnew = np.append(x_nonrelnew, x_nonrel[i])

d_nonrel = np.sqrt((x_nonrelnew-150)**2 + 166**2)


plt.figure('non rel ')
plt.title('non rel' )
plt.plot(d_nonrel, tof_nonrelnew, '.')
plt.grid()
plt.xlabel('distanza percorsa [cm]')
plt.ylabel('ToF [ns]')
plt.show()


v_nonrel = d_nonrel /(tof_nonrelnew + 27.99)
print(v_nonrel.mean(), v_nonrel.mean()/30)



beta_nonrel = d_nonrel /((tof_nonrelnew + 27.99)*30)

c1=TCanvas("Non rel", "Non rel")
#fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
#fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000) 
num_bin = 100

# Crea un istogramma con il numero di bin specificato
hist1=TH1F("V_nonrel", "V_nonrel; Beta; eventi", num_bin, -5, 5)
for i in range(len(beta_nonrel)):
	hist1.Fill(beta_nonrel[i])

#hist1.Fit("fit_function", "ILR")
#fit_function.Draw()
hist1.Draw()

c1.Draw()

input()


