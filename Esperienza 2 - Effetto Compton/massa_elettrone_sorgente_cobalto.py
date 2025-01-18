import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from convertitore import convertitore
from fit_sorgenti import fit_cobalto, fit_sodio, fit_cesio
from fit_sorgenti2 import fit_cobalto2, fit_sodio2, fit_cesio2
from calibrazione import calibrazione
from compton import compton_fit
from energie_compton import energie_compton
from compton2 import compton_fit2

#************************************************************************************************************************************************************************************************

# CALCOLO ERRORI ANDAMENTI SORGENTI E COMPTON PER VARIAZIONE RANGE, BINNING E BKG FUNCTION

#*************************************************************************************************************************************************************************************************

picco_Co1, err_picco_Co1, sigma_Co1, err_sigma_Co1, picco_Co2, err_picco_Co2, sigma_Co2, err_sigma_Co2 = plb.loadtxt('dati/andamento_cobalto.txt', unpack=True)
picco_Na1, err_picco_Na1, sigma_Na1, err_sigma_Na1, picco_Na2, err_picco_Na2, sigma_Na2, err_sigma_Na2 = plb.loadtxt('dati/andamento_sodio.txt', unpack=True)
picco_Cs, err_picco_Cs, sigma_Cs, err_sigma_Cs = plb.loadtxt('dati/andamento_cesio.txt', unpack=True)
picco_1, err_picco_1, sigma_1, err_sigma_1, picco_2, err_picco_2, sigma_2, err_sigma_2 = plb.loadtxt('dati/andamento_compton.txt', unpack=True)

media_Co1=np.mean(picco_Co1)
err_media_Co1=np.mean(err_picco_Co1)
media_Co2=np.mean(picco_Co2)
err_media_Co2=np.mean(err_picco_Co2)

media_Na1=np.mean(picco_Na1)
err_media_Na1=np.mean(err_picco_Na1)
media_Na2=np.mean(picco_Na2)
err_media_Na2=np.mean(err_picco_Na2)

media_Cs=np.mean(picco_Cs)
err_media_Cs=np.mean(err_picco_Cs)

media_1=np.mean(picco_1)
err_media_1=np.mean(err_picco_1)
media_2=np.mean(picco_2)
err_media_2=np.mean(err_picco_2)

err_Co1=(max(picco_Co1)-min(picco_Co1))/2 
err_Co2=(max(picco_Co2)-min(picco_Co2))/2 

err_Na1= (max(picco_Na1)-min(picco_Na1))/2 
err_Na2=(max(picco_Na2)-min(picco_Na2))/2 

err_Cs= (max(picco_Cs)-min(picco_Cs))/2 

err_1=(max(picco_1)-min(picco_1))/2 
err_2= (max(picco_2)-min(picco_2))/2 

dCo1=(err_Co1/media_Co1)
dCo2=(err_Co2/media_Co2)

dNa1=(err_Na1/media_Na1)
dNa2=(err_Na2/media_Na2)

dCs=(err_Cs/media_Cs)

dCompton1=(err_1/media_1)
dCompton2=(err_2/media_2)


#************************************************************************************************************************************************************************************************

# FIT SORGENTE COBALTO PER ENERGIE TABULATE

#*************************************************************************************************************************************************************************************************
cobalto_sorgente1=convertitore("03-21/calibrazione/inizio/cobalto_sorgente_esadecimale.dat")
cobalto_sorgente2=convertitore("03-21/calibrazione/fine/cobalto_sorgente_esadecimale.dat")

ch_tabA_1, ch_tabA_err_1, ch_tabB_1, ch_tabB_err_1=energie_compton(cobalto_sorgente1, 300)
ch_tabA_2, ch_tabA_err_2, ch_tabB_2, ch_tabB_err_2=energie_compton(cobalto_sorgente2, 300)

# Inizializzazione delle liste vuote per i valori ch e dch
ch_list_tab = []
dch_list_tab = []


# Ciclo for per creare ch e dch per ciascun campione
for i in range(1, 3):
    pippo = np.array([globals()[f"ch_tabA_{i}"], globals()[f"ch_tabB_{i}"]])
    pluto = np.array([globals()[f"ch_tabA_err_{i}"], globals()[f"ch_tabB_err_{i}"]])
    # Aggiungere i valori ch e dch alle rispettive liste
    ch_list_tab.append(pippo)
    dch_list_tab.append(pluto)

ch_tab=np.array(ch_list_tab)
dch_tab=np.array(dch_list_tab)


media_ch_tabA=np.mean(ch_tab[:, 0])
media_ch_tabB=np.mean(ch_tab[:, 1])
media_ch_tabA_err=np.sqrt(np.sum(dch_tab[:, 0]**2)/(len(dch_tab[:, 0])**2))
media_ch_tabB_err=np.sqrt(np.sum(dch_tab[:, 1]**2)/(len(dch_tab[:, 1])**2))


err_ch_tabA=np.sqrt( ((np.max(ch_tab[:, 0])-np.min(ch_tab[:,0]))/2)**2+media_ch_tabA_err**2+(dCompton1*media_ch_tabA)**2)
err_ch_tabB=np.sqrt( ((np.max(ch_tab[:, 1])-np.min(ch_tab[:,1]))/2)**2+media_ch_tabB_err**2+(dCompton2*media_ch_tabB)**2)

print("E iniziali in canali")
print(media_ch_tabA, err_ch_tabA)
print(media_ch_tabB, err_ch_tabB)

input()
#************************************************************************************************************************************************************************************************


E=np.array([1.173, 1.333])
ch=np.array([media_ch_tabA, media_ch_tabB])
dch=np.array([err_ch_tabA, err_ch_tabB])

c_cal = TCanvas("calibrazione", "calibrazione")
c_cal.SetGrid()

linear1 = TF1("linear", "pol1(0)", 0, 2)
linear1.SetParameters(216, 4586) 
parabola1 = TF1("parabola", "pol2(0)", 0, 2)
parabola1.SetParameters(-256,5695, -588) 



graph1 =TGraphErrors(2, E, ch, nullptr, dch)



# Imposta lo stile del grafico
graph1.SetLineColor(kBlack)
graph1.SetTitle("Calibrazione mediata")
graph1.SetMarkerStyle(20)
graph1.SetMarkerSize(1)
# Aggiungi le label agli assi
graph1.GetXaxis().SetTitle("E [MeV]")
graph1.GetYaxis().SetTitle("ch")	
# Effettua i fit e ottieni l'oggetto TFitResult
result_linear1 = graph1.Fit(linear1, "S")
result_parabola1 = graph1.Fit(parabola1, "S")	
# Disegna il grafico con i punti e le linee dei fit
graph1.Draw("AP")
# Disegna la funzione del primo fit
linear1.SetLineColor(kBlue)
linear1.Draw("same")	
# Disegna la funzione del secondo fit
parabola1.SetLineColor(kRed)
parabola1.Draw("same")

# Visualizza il canvas
c_cal.Draw()


#************************************************

# PARAMETRI FIT

#*************************************************

# Estrai i parametri del fit
m = linear1.GetParameter(1)
dm = linear1.GetParError(1)
q = linear1.GetParameter(0)
dq = linear1.GetParError(0)
c = parabola1.GetParameter(0)
dc = parabola1.GetParError(0)
b = parabola1.GetParameter(1)
db = parabola1.GetParError(1)
a = parabola1.GetParameter(2)
da = parabola1.GetParError(2)

# Ottieni le matrici di covarianza dai risultati del fit
cov_matrix_linear1 = result_linear1.GetCovarianceMatrix()
cov_matrix_parabola1 = result_parabola1.GetCovarianceMatrix()

# Stampa le matrici di covarianza
print("Matrice di covarianza per il fit lineare:")
cov_matrix_linear1.Print()
print("Matrice di covarianza per il fit parabolico:")
cov_matrix_parabola1.Print()



# Converti la matrice di covarianza in un array numpy
cov_linear = np.array([[cov_matrix_linear1(i, j) for j in range(cov_matrix_linear1.GetNcols())] for i in range(cov_matrix_linear1.GetNrows())])
cov_parabola = np.array([[cov_matrix_parabola1(i, j) for j in range(cov_matrix_parabola1.GetNcols())] for i in range(cov_matrix_parabola1.GetNrows())])



input()

#************************************************************************************************************************************************************************************************

# FIT COMPTON

#*************************************************************************************************************************************************************************************************

compton2= convertitore("03-21/compton/compton_15deg_2_esadecimale.dat")
compton= convertitore("03-21/compton/compton_15deg_esadecimale.dat")
 
# fit function double gauss+polynomial per 21 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4200, 6400)
fit_function_compton.SetParameters(180, 4914, 140, 211, 5496, 189, 328, 0.04, -1.37e-05)


#creo il canvas
c_compton = TCanvas("compton", "compton")

#decido i bin 
num_bin = 250
# Crea un istogramma con il numero di bin specificato
hist_compton=TH1F("compton", "Spettro compton", num_bin, 0, 8192)
i=0
# Riempie i bin con le altezze specificate
for i in range(len(compton)):
	hist_compton.Fill(compton[i])


for i in range(len(compton2)):
	hist_compton.Fill(compton2[i])

	
# Adattamento della funzione ai dati
hist_compton.Fit("fit_function_compton", "ILR")


hist_compton.Draw()
fit_function_compton.Draw("same")


#parametri fit
p0_compton = fit_function_compton.GetParameter(0)
p0_err_compton = fit_function_compton.GetParError(0)
p1_compton = fit_function_compton.GetParameter(1)
p1_err_compton = fit_function_compton.GetParError(1)
p2_compton = fit_function_compton.GetParameter(2)
p2_err_compton = fit_function_compton.GetParError(2)
p3_compton = fit_function_compton.GetParameter(3)
p3_err_compton = fit_function_compton.GetParError(3)
p4_compton = fit_function_compton.GetParameter(4)
p4_err_compton = fit_function_compton.GetParError(4)
p5_compton = fit_function_compton.GetParameter(5)
p5_err_compton = fit_function_compton.GetParError(5)
p6_compton = fit_function_compton.GetParameter(6)
p6_err_compton = fit_function_compton.GetParError(6)
p7_compton = fit_function_compton.GetParameter(7)
p7_err_compton = fit_function_compton.GetParError(7)
p8_compton = fit_function_compton.GetParameter(8)
p8_err_compton = fit_function_compton.GetParError(8)


p1_err_compton=np.sqrt( p1_err_compton**2+(dCompton1*p1_compton)**2)
p4_err_compton=np.sqrt( p4_err_compton**2+(dCompton2*p4_compton)**2)

c_compton.Draw()

print("E finali in canali")
print(p1_compton, p1_err_compton)
print(p4_compton, p4_err_compton)

input()

#*************************************************************************************************************************************************************************************

# MASSA ELETTRONE

#***************************************************************************************************************************************************************************************

def conversionelineare(ch, q, m):
	E_lin=(ch-q)/m
	return E_lin

def conversionepolinomiale(ch, a, b, c):
	E_pol=(-b+np.sqrt(b**2-4*a*(c-ch)))/(2*a)
	return E_pol
	
def E_lin_err(m, q, m_err, q_err, ch, ch_err, cov_mq):
    # Calcolo delle derivate parziali rispetto ai parametri
    dE_dm = -1 / (m**2)
    dE_dq = -1 / m
    dE_dch = 1 / m

    # Calcolo dell'errore utilizzando le derivate parziali
    E_lin_err = np.sqrt((dE_dm * m_err)**2 + (dE_dq * q_err)**2 + (dE_dch * ch_err)**2 + 2 * dE_dm * dE_dq * cov_mq)
    
    return E_lin_err
    

def E_pol_err(ch, ch_err, a, b, c, a_err, b_err, c_err, cov_ab, cov_ac, cov_bc):
    # Calcolo delle derivate parziali rispetto ai parametri
    dE_da = (1 / (2*a)) * ((-b + np.sqrt(b**2 - 4*a*(c - ch))) / (2*np.sqrt(b**2 - 4*a*(c - ch))))
    dE_db = (1 / (2*a)) * (1 / np.sqrt(b**2 - 4*a*(c - ch)))
    dE_dc = (1 / (2*a)) * (-1 / np.sqrt(b**2 - 4*a*(c - ch)))
    dE_dch = (1 / (2*a)) * (1 / np.sqrt(b**2 - 4*a*(c - ch)))

    # Calcolo dell'errore utilizzando le derivate parziali
    E_pol_err = np.sqrt((dE_da * a_err)**2 + (dE_db * b_err)**2 + (dE_dc * c_err)**2 + (dE_dch * ch_err)**2 + 2 * dE_da * dE_db * cov_ab + 2 * dE_da * dE_dc * cov_ac + 2 * dE_db * dE_dc * cov_bc)
    
    return E_pol_err


cov_mq=cov_linear[0,1]
cov_ac=cov_parabola[0,2]
cov_bc=cov_parabola[1,2]
cov_ab=cov_parabola[0,1]
	
#conversione picchi compton		
E_a_lin=conversionelineare(p1_compton, q, m)
E_b_lin=conversionelineare(p4_compton, q, m)
dE_a_lin=E_lin_err(m, q, dm, dq, p1_compton, p1_err_compton, cov_mq)
dE_b_lin=E_lin_err(m, q, dm, dq, p4_compton, p4_err_compton, cov_mq)

E_a_pol=conversionepolinomiale(p1_compton, a, b, c)
E_b_pol=conversionepolinomiale(p4_compton, a, b, c)
dE_a_pol=E_pol_err(p1_compton, p1_err_compton, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)
dE_b_pol=E_pol_err(p4_compton, p4_err_compton, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)


#conversione E "tabulate" sorgente
E_in_a_lin=conversionelineare(media_ch_tabA, q, m)
E_in_b_lin=conversionelineare(media_ch_tabB, q, m)
E_in_a_err_lin=E_lin_err(m, q, dm, dq, media_ch_tabA, err_ch_tabA, cov_mq)
E_in_b_err_lin=E_lin_err(m, q, dm, dq, media_ch_tabB, err_ch_tabB, cov_mq)

E_in_a_pol=conversionepolinomiale(media_ch_tabA, a, b, c)
E_in_b_pol=conversionepolinomiale(media_ch_tabB, a, b, c)
E_in_a_err_pol=E_pol_err(media_ch_tabA, err_ch_tabA, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)
E_in_b_err_pol=E_pol_err(media_ch_tabB, err_ch_tabB, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)



print('\n')
print("E finali lin, err in perc")
print(E_a_lin, dE_a_lin, (dE_a_lin/E_a_lin)*100)
print(E_b_lin, dE_b_lin, (dE_b_lin/E_b_lin)*100)
print("E finali pol, err in perc")
print(E_a_pol, dE_a_pol, (dE_a_pol/E_a_pol)*100 )
print(E_b_pol, dE_b_pol, (dE_b_pol/E_b_pol)*100)
print('\n')


#*****************************************************************

#E iniziali tabulate



print('\n')
print("E iniz lin")
print(E_in_a_lin, E_in_a_err_lin, (E_in_a_err_lin/E_in_a_lin)*100)
print(E_in_b_lin, E_in_b_err_lin, (E_in_b_err_lin/E_in_b_lin)*100)
print("E iniz pol")
print(E_in_a_pol, E_in_a_err_pol, (E_in_a_err_pol/E_in_a_pol)*100)
print(E_in_b_pol, E_in_b_err_pol, (E_in_b_err_pol/E_in_b_pol)*100)
print('\n')


#******************************************************************



def m_E(E, E_in, theta):
	return (E*E_in*(1-np.cos(theta)))/(E_in-E)


	
#senza l'errore su E iniziale, si usa E tabulato	
def m_E_err(E, E_in, E_err, theta,  theta_err):
    
	d_m_E_dE = (E_in * (1 - np.cos(theta))) / (E_in - E) + (E * E_in * (1 - np.cos(theta))) / ((E_in - E) ** 2)
	d_m_E_dtheta = (E*E_in*np.sin(theta))/(E-E_in)
	m_E_err = np.sqrt((d_m_E_dE * E_err) ** 2 + (d_m_E_dtheta * theta_err) ** 2)

	return m_E_err



m_e1a_lin=m_E(E_a_lin, 1.173, 15*np.pi/180)
m_e1b_lin=m_E(E_b_lin, 1.333, 15*np.pi/180)
dm_e1a_lin=m_E_err(E_a_lin, 1.173,  dE_a_lin, 15*np.pi/180, np.pi/180)
dm_e1b_lin=m_E_err(E_b_lin, 1.333,  dE_b_lin, 15*np.pi/180, np.pi/180)

m_e1a_pol=m_E(E_a_pol, 1.173, 15*np.pi/180)
m_e1b_pol=m_E(E_b_pol, 1.333, 15*np.pi/180)
dm_e1a_pol=m_E_err(E_a_pol, 1.173,  dE_a_pol, 15*np.pi/180, np.pi/180)
dm_e1b_pol=m_E_err(E_b_pol, 1.333,  dE_b_pol, 15*np.pi/180, np.pi/180)



"""	
#includendo l'errore su l'energia iniziale, trovata calibrando la sorgente grande di cobalto
def m_E_err(E, E_in, E_err, E_in_err, theta,  theta_err):
    
	d_m_E_dE = (E_in * (1 - np.cos(theta))) / (E_in - E) + (E * E_in * (1 - np.cos(theta))) / ((E_in - E) ** 2)
	d_m_E_dE_in = (E * (1 - np.cos(theta))) / (E_in - E) - (E * E_in * (1 - np.cos(theta))) / ((E_in - E) ** 2)
	d_m_E_dtheta = (E*E_in*np.sin(theta))/(E-E_in)
	m_E_err = np.sqrt((d_m_E_dE * E_err) ** 2 + (d_m_E_dtheta * theta_err) ** 2+(d_m_E_dE_in*E_in_err)**2)

	return m_E_err	

m_e1a_lin=m_E(E_a_lin, E_in_a_lin, 15*np.pi/180)
m_e1b_lin=m_E(E_b_lin, E_in_b_lin, 15*np.pi/180)
dm_e1a_lin=m_E_err(E_a_lin, E_in_a_lin,  dE_a_lin, E_in_a_err_lin, 15*np.pi/180, np.pi/180)
dm_e1b_lin=m_E_err(E_b_lin, E_in_b_lin,  dE_b_lin, E_in_b_err_lin, 15*np.pi/180, np.pi/180)

m_e1a_pol=m_E(E_a_pol, E_in_a_pol, 15*np.pi/180)
m_e1b_pol=m_E(E_b_pol, E_in_b_pol, 15*np.pi/180)
dm_e1a_pol=m_E_err(E_a_pol, E_in_a_pol,  dE_a_pol, E_in_a_err_pol, 15*np.pi/180, np.pi/180)
dm_e1b_pol=m_E_err(E_b_pol, E_in_b_pol,  dE_b_pol, E_in_b_err_pol, 15*np.pi/180, np.pi/180)
"""




print(m_e1a_lin, dm_e1a_lin )
print(m_e1b_lin, dm_e1b_lin)
print('\n')
print(m_e1a_pol, dm_e1a_pol)
print(m_e1b_pol, dm_e1b_pol)


"""

masse = np.array([
    ['#conversione_lineare', '#primo_picco', '#secondo_picco'],
    [m_e1a_lin*1000, dm_e1a_lin*1000, ''],
    [m_e1b_lin*1000, dm_e1b_lin*1000, ''],
    ['#conversione_quadratica', '#primo_picco', '#secondo_picco'],
    [m_e1a_pol*1000, dm_e1a_pol*1000, ''],
    [m_e1b_pol*1000, dm_e1b_pol*1000, '']
], dtype=object)

# Save the array to a text file
np.savetxt('masse elettroni/masse_elettroni_19e20marzo_mediate.txt', masse, fmt='%s')
"""
