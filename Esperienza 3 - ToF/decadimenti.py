import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d


#*****************************************************************************************************************************************************************************************************

# DATI DRS

# Apri il file di root contenente il tuo albero
file = TFile.Open("05-23/Decay_05-23_DRS_2.root", "UPDATE")

# Ottieni l'albero dal file
tree_unclean = file.Get("rec")

# Numero di eventi nel vecchio albero
N = tree_unclean.GetEntries()
print(N)


# Indici degli eventi da eliminare
#indices_to_remove = {346, 350, 397, 414, 451, 452, 453, 623, 681, 810, 864}  # per il 21-05
#indices_to_remove = {14, 192, 263, 274, 316, 328, 331, (357+7), (439+8), (579+9), (587+10), (589+11), (725+12), (764+13), (902+14), (946+15)}  # per il 17-05
#indices_to_remove={None} #per il 22-05 parte 1
#indices_to_remove={0} #per il 22-05 parte 2
indices_to_remove={None} #per il 23-05 parte 1
#indices_to_remove={None} #per il 23-05 parte 2

# Crea una copia della struttura del vecchio albero, senza dati
tree = tree_unclean.CloneTree(0)

# Itera su ogni evento del vecchio albero e copia solo quelli che non vuoi eliminare
for i in range(N):
    if i not in indices_to_remove:  # Salta gli eventi con indici specificati
        tree_unclean.GetEntry(i)
        tree.Fill()



# Stampa il numero di eventi nel nuovo albero
N = tree.GetEntries()
print(f"Number of entries in the new tree: {N}")




    


timestamp_DRS=np.array([], dtype=int)
timestamp_DRS_string=np.zeros(N, dtype='U26')

# Itera su ogni evento nel tuo albero
for i in range(N):

	
	# Ottieni l'evento i-esimo
	tree.GetEntry(i)

	#tree.Scan("day:hours:minutes:seconds:mseconds")

	
	day_event = int(tree.GetLeaf("day").GetValue())
	hour_event= int(tree.GetLeaf("hours").GetValue())
	minute_event=int(tree.GetLeaf("minutes").GetValue())
	second_event=int(tree.GetLeaf("seconds").GetValue())
	millisecond_event=int(tree.GetLeaf("mseconds").GetValue())


	
	timestamp_DRS_string[i] = f'2024-05-{day_event:02d} {hour_event:02d}:{minute_event:02d}:{second_event:02d}.{millisecond_event*1000:06d}'
	timestamp_DRS = np.append(timestamp_DRS, (day_event*86400000+hour_event*3600000+ minute_event*60000+ second_event*1000+ millisecond_event))
		
#******************************************************************************************************************************************************************************************************






#*****************************************************************************************************************************************************************************************************

# DATI FPGA

data = np.loadtxt('05-23/Decay_05-23_2.dat', delimiter='\t', dtype={'names': ('col1', 'col2', 'col3'), 'formats': ('i4', 'f8', 'U26')})
ch=data['col1']
timestamp_FPGA_string=data['col3']
t=data['col2']

p=0
for i in range(len(ch)-1):
	if((ch[i]==2) & (ch[i+1]==2)):
		print('indici canali 2 consecutivi', i)
	if((ch[i]==1) & (ch[i+1]==2) & (t[i+1]-t[i]<0) ):
		print('timestamp e indice di START STOP con dT negativo:', i, timestamp_FPGA_string[i],timestamp_FPGA_string[i+1])
	if((ch[i]==1) & (ch[i+1]==2) & (abs(t[i+1]-t[i])>10e-5) ):
		print('START e STOP non compatibili all\'indice:', i)


def time_FPGA(timestamp_fpga):
	try:
		# Prova a convertire il timestamp con i microsecondi
		date_object = datetime.strptime(timestamp_fpga, '%Y-%m-%d %H:%M:%S.%f')
	except ValueError:
        	# Se la conversione con i microsecondi fallisce, prova senza di essi
		date_object = datetime.strptime(timestamp_fpga, '%Y-%m-%d %H:%M:%S')

	time_fpga = (date_object.day*86400000)+(date_object.hour * 3600000) + (date_object.minute * 60000) + (date_object.second * 1000)
    
	# Controlla se ci sono microsecondi
	if hasattr(date_object, 'microsecond'):
		time_fpga += int(date_object.microsecond / 1000)
    
	return time_fpga

# Applica la funzione alla terza colonna
timestamp_FPGA = np.array([time_FPGA(row['col3']) for row in data])

#******************************************************************************************************************************************************************************************************






#******************************************************************************************************************************************************************************************************

# LETTURA INDICI

index=np.array([], dtype=int)
index_ch2=np.array([], dtype=int)
index_false_start=np.array([], dtype=int)

a=0
b=0
for i in range(len(ch)):
	if(ch[i]==1):
		b=b+1
	if(ch[i]==2):
		index_ch2=np.append(index_ch2, i)

timestamp_FPGA = np.delete(timestamp_FPGA, index_ch2)
timestamp_FPGA_string=np.delete(timestamp_FPGA_string, index_ch2)
ch_clean=np.delete(ch, index_ch2)
t_clean=np.delete(t, index_ch2)
		

for i in range(1, len(ch)):
	if((ch[i]==2) & (ch[i-1]==1) ):
		a=a+1
		index=np.append(index, i-a)
		
print('Il numero di STOP è:', a)

#******************************************************************************************************************************************************************************************************







#****************************************************************************************************************************************************************************************

# SALVATAGGIO FILE DI TESTO

# Crea un array bidimensionale riempito con stringhe vuote

max_len = max(len(timestamp_DRS_string), len(timestamp_FPGA_string))
min_len = min(len(timestamp_DRS), len(timestamp_FPGA))


diff_timestamp=timestamp_DRS[:min_len]-timestamp_FPGA[:min_len]

# Alternativa: puoi salvare i dati senza conversione in stringhe, usando un array strutturato
structured_array = np.zeros((max_len,), dtype=[('DRS', 'U26'), ('FPGA', 'U26'), ('Diff', 'f8')])
structured_array['DRS'][:len(timestamp_DRS_string)] = timestamp_DRS_string
structured_array['FPGA'][:len(timestamp_FPGA_string)] = timestamp_FPGA_string
structured_array['Diff'][:len(diff_timestamp)] = diff_timestamp

# Salva l'array strutturato con intestazione
header = "#DRS				#FPGA				#Diff"
#np.savetxt('05-22/timestamps_output_2.txt', structured_array, fmt='%s\t%s\t%.6f', header=header, comments='')

ch1_timestampFPGA = np.zeros((len(timestamp_FPGA_string),), dtype=[('CH', 'd'), ('T', 'f8'), ('timestamp', 'U26')])
ch1_timestampFPGA['CH'][:len(timestamp_FPGA_string)] = ch_clean
ch1_timestampFPGA['T'][:len(timestamp_FPGA_string)] = t_clean
ch1_timestampFPGA['timestamp'][:len(timestamp_FPGA_string)] = timestamp_FPGA_string

header_new="#CH	#T		#Timestamp"
#np.savetxt('05-22/timestamps_ch1_FPGA_2.txt', ch1_timestampFPGA, fmt='%d\t%.8f\t%s', header=header_new, comments='')

#Controllo per verificare gli indici problematici, gli indici si riferiscono ai file con i ch2 cancellati
index_problem=np.where(abs(diff_timestamp)>=1150)

if(len(index_problem)>0):
	print('numero di indici corrispondenti a diff. timestamp >1000', index_problem)
	
if(len(timestamp_FPGA_string)==len(timestamp_FPGA)==len(ch_clean)==len(t_clean)):
	print('numero start FPGA:', len(timestamp_FPGA))

if(len(timestamp_DRS_string)==len(timestamp_DRS)):	
	print('numero start DRS:', len(timestamp_DRS_string))

if(len(timestamp_FPGA_string)==len(timestamp_FPGA)==len(ch_clean)==len(t_clean)==len(timestamp_DRS_string)==len(timestamp_DRS)):
	print('DRS e FPGA sono allineati')
	

#****************************************************************************************************************************************************************************************






"""
#****************************************************************************************************************************************************************************************

# ISTOGRAMMA








c=TCanvas("#Delta timestamp DRS-FPGA", "#Delta timestamp DRS-FPGA")

 
num_bin = int(10**6)

# Crea un istogramma con il numero di bin specificato
hist=TH1F("#Delta timestamp DRS-FPGA", "#Delta timestamp DRS-FPGA", num_bin, -40, 120)
for i in range(len(diff_timestamp)):
	hist.Fill(diff_timestamp[i])


hist.Draw()



c.Draw()

input()

#****************************************************************************************************************************************************************************************

"""




#****************************************************************************************************************************************************************************************

# CALCOLO DEI TEMPI

# Inizializza gli array bidimensionali per w1 e t1, composti da N righe e 1024 colonne corrispondenti alle N righe
w0 = np.zeros((len(index), 1024))
t0 = np.zeros((len(index), 1024))
w1 = np.zeros((len(index), 1024))
t1 = np.zeros((len(index), 1024))
w2 = np.zeros((len(index), 1024))
t2 = np.zeros((len(index), 1024))


k=0
# Itera su ogni evento nel tuo albero
for i, el in enumerate(index):
	
	# Ottieni l'evento i-esimo
	tree.GetEntry(el)
    
	# Ottieni i valori dei rami w1 e t1 per l'evento i-esimo
	w0_event = tree.w0
	t0_event = tree.t0
	w1_event = tree.w1
	t1_event = tree.t1
	w2_event = tree.w2
	t2_event = tree.t2
    
	# Riempi l'array bidimensionale con i valori dell'evento i-esimo
	for j in range(1024):
		w0[i][j] = w0_event[j]
		t0[i][j] = t0_event[j]
		w1[i][j] = w1_event[j]
		t1[i][j] = t1_event[j]
		w2[i][j] = w2_event[j]
		t2[i][j] = t2_event[j]
	



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




for i in range(len(index)):
	
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
					


			t_pmt1=np.append(t_pmt1, t0_soglia)
			t_pmt2=np.append(t_pmt2, t1_soglia)
			t_pmt3=np.append(t_pmt3, t2_soglia)
			


if(len(t_pmt1)==len(t_pmt1)==len(t_pmt1)):
	print('Il numero di STOP filtrati è:', len(t_pmt1))



"""
np.savetxt('05-22/T1_ToF_decay_2.txt', t_pmt1)
np.savetxt('05-22/T2_ToF_decay_2.txt', t_pmt2)
np.savetxt('05-22/T3_ToF_decay_2.txt', t_pmt3)
"""

#****************************************************************************************************************************************************************************************





"""
#***************************************************************************************************************************************************************************

# CONTROLLO GRAFICO
for i in range(N):
	canvas = TCanvas("canvas", "Segnale PMT01", 800, 600)
		
	# Estrai la riga da t e w
	riga_t = t0[i]  
	riga_w = w0[i]


	graph = TGraph(1024)

	# Riempire il grafico con i dati
	for j in range(1024):
		graph.SetPoint(j, riga_t[j], riga_w[j])




	#graph.SetMarkerStyle(kFullCircle)
	#graph.SetMarkerSize(0.8)
	#graph.SetMarkerColor(kBlue)
	graph.SetTitle("Segnale PMT01")
	graph.GetXaxis().SetTitle("t [ns]")
	graph.GetYaxis().SetTitle("V [V]")
	graph.Draw("ACP")

	canvas.Draw()




	canvas1 = TCanvas("canvas1", "Segnale PMT02", 800, 600)

	# Estrai la da t e w
	riga_t1 = t1[i]  
	riga_w1 = w1[i]


	graph1 = TGraph(1024)


	for j in range(1024):
		graph1.SetPoint(j, riga_t1[j], riga_w1[j])



	graph1.Draw("ACP")  # "AP" per disegnare i punti con gli assi


	graph1.GetXaxis().SetTitle("t")
	graph1.GetYaxis().SetTitle("w")


	canvas1.Draw()
	
	canvas2 = TCanvas("canvas2", "Segnale PMT03", 800, 600)

	# Estrai la da t e w
	riga_t2 = t2[i]  
	riga_w2 = w2[i]


	graph2 = TGraph(1024)


	for j in range(1024):
		graph2.SetPoint(j, riga_t2[j], riga_w2[j])



	graph2.Draw("ACP")  # "AP" per disegnare i punti con gli assi


	graph2.GetXaxis().SetTitle("t")
	graph2.GetYaxis().SetTitle("w")


	canvas2.Draw()

	input()	
#***************************************************************************************************************************************************************************
"""




















		
		
