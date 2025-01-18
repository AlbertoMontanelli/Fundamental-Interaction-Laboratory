import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d


# Apri il file di root contenente il tuo albero
file = TFile.Open("04-18/2.5cm_20000ev.root")

# Ottieni l'albero dal file
tree = file.Get("rec")

# numero di eventi che ci sono
N=tree.GetEntries()


# Inizializza gli array bidimensionali per w1 e t1, composti da N righe e 1024 colonne corrispondenti alle N righe
w0 = np.zeros((N, 1024))
t0 = np.zeros((N, 1024))
w1 = np.zeros((N, 1024))
t1 = np.zeros((N, 1024))
w2 = np.zeros((N, 1024))
t2 = np.zeros((N, 1024))

# Itera su ogni evento nel tuo albero
for i in range(N):
	# Ottieni l'evento i-esimo
	tree.GetEntry(i)
    
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




#***************************************************************************************************************************************************************************

# CONTROLLO GRAFICO
for i in range(2403, 2404):
	canvas = TCanvas("canvas", "Segnale PMT01", 800, 600)
		
	# Estrai la riga da t e w
	riga_t = t0[i]  
	riga_w = w0[i]


	graph = TGraph(1024)
	#graph1=TGraph(1024)
	
	# Riempire il grafico con i dati
	for j in range(1024):
		graph.SetPoint(j, riga_t[j], riga_w[j])
		#graph1.SetPoint(j, riga_t[j], riga_w[j])



	#graph.SetMarkerStyle(kFullCircle)
	#graph.SetMarkerSize(0.8)
	graph.SetMarkerColor(kRed)
	graph.SetTitle("Segnale PMT01")
	graph.GetXaxis().SetTitle("t [ns]")
	graph.GetYaxis().SetTitle("V [V]")

	graph.SetLineWidth(1)
	graph.SetMarkerStyle(kFullCircle)
	graph.SetMarkerSize(1)	
	graph.Draw("ACP")
	#graph1.Draw("AP")
	
	canvas.Draw()
	
	input()


"""

	canvas1 = TCanvas("canvas1", "Grafico TGraph1", 800, 600)

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
	
	canvas2 = TCanvas("canvas2", "Grafico TGraph2", 800, 600)

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
"""
	
	
#***************************************************************************************************************************************************************************


"""

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
			
			print(i)
			if(len(t0_soglia)>=1):
				print('t0')
			if(len(t1_soglia)>=1):
				print('t1')
			if(len(t2_soglia)>=1):
				print('t2')
			riga_t = t2[i]  
			riga_w = w2[i]

			canvas = TCanvas("canvas", "Grafico TGraph", 800, 600)
			graph = TGraph(1024)
				

				

			
			graph_puntosoglia=TGraph(1)
			graph_puntosoglia.SetPoint(1, t2[i][j2], w2[i][j2])
			graph_puntosoglia.SetMarkerStyle(kFullCircle)
			graph_puntosoglia.SetMarkerSize(1)
			graph_puntosoglia.SetMarkerColor(kGreen)

			retta_soglia0=TLine(19, 0.4*min_w2, 25, 0.4*min_w2)
			retta_soglia0.SetLineWidth(1)	
						 
			retta_control_0a=TLine( t2[i][j2]+t2_control, min_w2, t2[i][j2]+t2_control, max(w2[i]))
			retta_control_0b=TLine( t2[i][j2], min_w2, t2[i][j2], max(w2[i]))				
			retta_control_0a.SetLineColor(kGray)
			retta_control_0a.SetLineWidth(1)
			retta_control_0b.SetLineColor(kGray)
			retta_control_0b.SetLineWidth(1)



			for j in range(1024):
    				graph.SetPoint(j, riga_t[j], riga_w[j])


			graph.GetXaxis().SetTitle("t [ns]")
			graph.GetYaxis().SetTitle("V [V]")
			graph.SetTitle('Segnale PMT03')	
			graph.SetMarkerStyle(kFullCircle)
			graph.SetMarkerSize(1)
			graph.Draw("ACP")


			graph_puntosoglia.Draw("P")
			retta_soglia0.Draw()
			retta_control_0a.Draw()
			retta_control_0b.Draw()
			canvas.Draw()
			input()
			
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

"""					
			
"""
			riga_t = t0[i]  
			riga_w = w0[i]

			canvas = TCanvas("canvas", "Grafico TGraph", 800, 600)
			graph = TGraph(1024)
				
			retta_0=TGraph(len(y0_linear))
			for m in range(len(y0_linear)):
				retta_0.SetPoint(m, x0_new[m], y0_linear[m])
				retta_0.SetMarkerColor(kOrange)	

				
			graph_t0_soglia = TGraph()
			graph_t0_soglia.SetPoint(0, t0_soglia, f0(t0_soglia))  
			graph_t0_soglia.SetMarkerStyle(kFullCircle)
			graph_t0_soglia.SetMarkerSize(1)
			graph_t0_soglia.SetMarkerColor(kBlue)

			graph_t0=TGraph(5)
			for p in range(0, 5):
				graph_t0.SetPoint(p, t0[i][j0-2+p], w0[i][j0-2+p])
			graph_t0.SetMarkerStyle(kFullCircle)
			graph_t0.SetMarkerSize(1)
			graph_t0.SetMarkerColor(kRed)
			
			graph_puntosoglia=TGraph(1)
			graph_puntosoglia.SetPoint(1, t0[i][j0], w0[i][j0])
			graph_puntosoglia.SetMarkerStyle(kFullCircle)
			graph_puntosoglia.SetMarkerSize(1)
			graph_puntosoglia.SetMarkerColor(kGreen)

			retta_soglia0=TLine(35, 0.4*min_w0, 44, 0.4*min_w0)
			retta_soglia0.SetLineWidth(1)	
						 
			retta_control_0a=TLine( t0[i][j0]+t0_control, min_w0, t0[i][j0]+t0_control, max(w0[i]))
			retta_control_0b=TLine( t0[i][j0], min_w0, t0[i][j0], max(w0[i]))				
			retta_control_0a.SetLineColor(kGray)
			retta_control_0a.SetLineWidth(1)
			retta_control_0b.SetLineColor(kGray)
			retta_control_0b.SetLineWidth(1)



			for j in range(1024):
    				graph.SetPoint(j, riga_t[j], riga_w[j])


			graph.GetXaxis().SetTitle("t [ns]")
			graph.GetYaxis().SetTitle("V [V]")
			graph.SetTitle('Segnale PMT01')	
			graph.SetMarkerStyle(kFullCircle)
			graph.SetMarkerSize(1)
			graph.Draw("ACP")
			retta_0.Draw("CP")  
			graph_t0_soglia.Draw("P")
			graph_t0.Draw("P") 
			graph_puntosoglia.Draw("P")
			retta_soglia0.Draw()
			retta_control_0a.Draw()
			retta_control_0b.Draw()
			canvas.Draw()
			input()
"""


		
"""		
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
			dT_slope_1e2=np.append(dT_slope_1e2, t1_soglia-t2_soglia)
			dT_slope_0e2=np.append(dT_slope_0e2, t0_soglia-t2_soglia)

			t_pmt1=np.append(t_pmt1, t0_soglia)
			t_pmt2=np.append(t_pmt2, t1_soglia)
			t_pmt3=np.append(t_pmt3, t2_soglia)


print(len(t_pmt1))
print(len(t_pmt2))
print(len(t_pmt3))
print('\n')
print(len(dT_slope_0e1))
print(len(dT_slope_1e2))
print(len(dT_slope_0e2))


np.savetxt('05-16/T1_ToF_decay.txt', t_pmt1)
np.savetxt('05-16/T2_ToF_decay.txt', t_pmt2)
np.savetxt('05-16/T3_ToF_decay.txt', t_pmt3)



np.savetxt("05-02/dT1e2_180cm.txt", dT_slope_0e1)
np.savetxt("05-02/dT2e3_180cm.txt", dT_slope_1e2)
np.savetxt("05-02/dT1e3_180cm.txt", dT_slope_0e2)







c1=TCanvas("#Delta T tra PMT01 e PMT02", "#Delta T tra PMT01 e PMT02")

 
num_bin = 200

# Crea un istogramma con il numero di bin specificato
hist1=TH1F("#Delta T tra PMT01 e PMT02", "#Delta T tra PMT01 e PMT02", num_bin, -25, 25)
for i in range(len(dT_slope_0e1)):
	hist1.Fill(dT_slope_0e1[i])


hist1.Draw()

c1.Draw()

input()


"""

			
"""
			if ( -14.3<=(t0_soglia-t1_soglia)<=-13.3):
				
				print('soglia per t0=', 0.4*min_w0)
				print('t0 e w0 ricavati=', t0_soglia, f0(t0_soglia))
				print('t0 e w0 reali=', t0[i][j0], w0[i][j0])
				print('w0-1 e w0+1 erano=' ,w0[i][j0-1], w0[i][j0+1])
				print('t0-1 e t0+1 erano=', t0[i][j0-1], t0[i][j0+1])
				
				riga_t = t0[i]  
				riga_w = w0[i]

				canvas = TCanvas("canvas", "Grafico TGraph", 800, 600)
				graph = TGraph(1024)
				
				retta_0=TGraph(len(y_linear0))
				for m in range(len(y_linear0)):
					retta_0.SetPoint(m, x_new0[m], y_linear0[m])
				retta_0.SetMarkerColor(kOrange)	

				
				graph_t0_soglia = TGraph()
				graph_t0_soglia.SetPoint(0, t0_soglia, f0(t0_soglia))  
				graph_t0_soglia.SetMarkerStyle(kFullCircle)
				graph_t0_soglia.SetMarkerSize(1)
				graph_t0_soglia.SetMarkerColor(kBlue)

				graph_t0=TGraph(5)
				for p in range(0, 7):
					graph_t0.SetPoint(p, t0[i][j0-3+p], w0[i][j0-3+p])
					print(t0[i][j0-3+p]-t0[i][j0-4+p])
				graph_t0.SetMarkerStyle(kFullCircle)
				graph_t0.SetMarkerSize(1)
				graph_t0.SetMarkerColor(kRed)

				retta_soglia0=TLine(0, 0.4*min_w0, max(riga_t), 0.4*min_w0)
				 
				retta_control_0a=TLine( t0[i][j0]+t0_control, min_w0, t0[i][j0]+t0_control, max(w0[i]))
				retta_control_0b=TLine( t0[i][j0], min_w0, t0[i][j0], max(w0[i]))				
				retta_control_0a.SetLineColor(kGreen)
				retta_control_0a.SetLineWidth(1)
				retta_control_0b.SetLineColor(kGreen)
				retta_control_0b.SetLineWidth(1)



				for j in range(1024):
    					graph.SetPoint(j, riga_t[j], riga_w[j])


				
				graph.SetMarkerStyle(kFullCircle)
				graph.SetMarkerSize(0.5)
				graph.Draw("AP")
				retta_0.Draw("CP")  
				graph_t0_soglia.Draw("P")
				graph_t0.Draw("P") 
				retta_soglia0.Draw()
				retta_control_0a.Draw()
				retta_control_0b.Draw()
				canvas.Draw()


				riga_t1 = t1[i]  
				riga_w1 = w1[i]
				
				print('\n')
				print('soglia per t1=', 0.4*min_w1)
				print('t1 e w1 ricavati=', t1_soglia, f1(t1_soglia))
				print('t1 e w1 reali=', t1[i][j1], w1[i][j1])
				print('w1-1 e w1+1 erano=' ,w1[i][j1-1], w1[i][j1+1])
				print('t1-1 e t1+1 erano=', t1[i][j1-1], t1[i][j1+1])
								
				canvas1 = TCanvas("canvas1", "Grafico TGraph1", 800, 600)
				graph1 = TGraph(1024)

				retta_1=TGraph(len(y_linear1))
				for m in range(len(y_linear1)):
					retta_1.SetPoint(m, x_new1[m], y_linear1[m])
				retta_1.SetMarkerColor(kOrange)
				
				for j in range(1024):
    					graph1.SetPoint(j, riga_t1[j], riga_w1[j])

				graph_t1_soglia = TGraph()
				graph_t1_soglia.SetPoint(0, t1_soglia, f1(t1_soglia))  
				graph_t1_soglia.SetMarkerStyle(kFullCircle)
				graph_t1_soglia.SetMarkerSize(1)
				graph_t1_soglia.SetMarkerColor(kBlue)

				graph_t1=TGraph()
				for p in range(0, 7):
					graph_t1.SetPoint(p, t1[i][j1-3+p], w1[i][j1-3+p])
					print(t1[i][j1-3+p]-t1[i][j1-4+p])
				graph_t1.SetMarkerStyle(kFullCircle)
				graph_t1.SetMarkerSize(1)
				graph_t1.SetMarkerColor(kRed)

				retta_soglia1=TLine(0, 0.4*min_w1, max(riga_t), 0.4*min_w1)
				 
				retta_control_1a=TLine( t1[i][j1]+t1_control, min_w1, t1[i][j1]+t1_control, max(w1[i]))
				retta_control_1b=TLine( t1[i][j1], min_w1, t1[i][j1], max(w1[i]))
				retta_control_1a.SetLineColor(kGreen)
				retta_control_1a.SetLineWidth(1)
				retta_control_1b.SetLineColor(kGreen)
				retta_control_1b.SetLineWidth(1)				
				
				graph1.SetMarkerStyle(kFullCircle)
				graph1.SetMarkerSize(0.5)
				graph1.Draw("AP")
				retta_1.Draw("CP")
				graph_t1_soglia.Draw("P")
				graph_t1.Draw("P") 
				retta_soglia1.Draw()
				retta_control_1a.Draw()
				retta_control_1b.Draw()

				canvas1.Draw()

				input()
"""









