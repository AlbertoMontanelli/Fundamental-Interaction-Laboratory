import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from convertitore import convertitore
from compton import compton_fit
from energie_compton import energie_compton
from compton2 import compton_fit2
#************************************************************************************************************************************************************************************************

# CALCOLO ERRORI ANDAMENTO COMPTON PER VARIAZIONE RANGE, BINNING E BKG FUNCTION

#*************************************************************************************************************************************************************************************************

picco_1, err_picco_1, sigma_1, err_sigma_1, picco_2, err_picco_2, sigma_2, err_sigma_2 = plb.loadtxt('dati/andamento_compton.txt', unpack=True)

media_1=np.mean(picco_1)
err_media_1=np.mean(err_picco_1)
media_2=np.mean(picco_2)
err_media_2=np.mean(err_picco_2) 

err_1=(max(picco_1)-min(picco_1))/2 
err_2= (max(picco_2)-min(picco_2))/2 


dCompton1=(err_1/media_1)
dCompton2=(err_2/media_2)

"""
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

"""


#************************************************************************************************************************************************************************************************

# FIT COMPTON

#*************************************************************************************************************************************************************************************************

#compton= convertitore("03-05/compton/compton_15deg_35cm.dat")
#compton2= convertitore("03-19/compton/compton_78deg-2_esadecimale.dat")
#compton= convertitore("03-07/compton/compton_18deg_35cm_esadecimale.dat")
#compton= convertitore("03-13/compton/compton_113deg_35cm_esadecimale.dat") 
compton= convertitore("03-19/compton/compton_78deg-1_esadecimale.dat")
compton2= convertitore("03-19/compton/compton_78deg-2_esadecimale.dat")
#compton= convertitore("03-19/compton/compton_74deg_3_esadecimale.dat")
#compton= convertitore("03-20/compton/compton_22deg_esadecimale.dat")
#compton= convertitore("03-21/compton/compton_15deg_esadecimale.dat")
#compton2= convertitore("03-21/compton/compton_15deg_2_esadecimale.dat")



"""
# fit function double gauss+polynomial per 7 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4600, 6210)
fit_function_compton.SetParameters(1200, 5584, 132, 2000, 4700, 143, -200,0.26, -3.5e-05 )


# fit function double gauss+polynomial per 12e13 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4600, 6210)
fit_function_compton.SetParameters(1200, 5584, 132, 2000, 4700, 143, -200,0.26, -3.5e-05 )

# fit function double gauss+polynomial per 13 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 5050, 6710)
fit_function_compton.SetParameters(710, 5584, 132, 725, 6308, 143, 13461, -3.85, 0.0002762 ) 
"""
# fit function double gauss+polynomial per 19 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol1(6)", 4500, 6430)
fit_function_compton.SetParameters(100, 5057.5, 185.392,  107.1,  5660.39, 171,  351, -0.05179  )
fit_function_compton.SetLineWidth(1)
"""
# fit function double gauss+polynomial per 19e20 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4600, 6200)
fit_function_compton.SetParameters(1500, 5057.5, 185.392,  1427.1,  5628.39, 205.4,    7916,   -2.128374, 0.000140811  )

# fit function double gauss+polynomial per 20e21 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4000, 6000)
fit_function_compton.SetParameters(1818, 4914, 210, 1467, 5496, 189, 5783, -1.19, 4.2e-05 )

# fit function double gauss+polynomial per 21 marzo
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4200, 6400)
fit_function_compton.SetParameters(180, 4914, 140, 211, 5496, 189, 328, 0.04, -1.37e-05)
"""


#creo il canvas
c_compton = TCanvas("compton", "Spettro Compton 19 Marzo")

#decido i bin 
num_bin = 150
# Crea un istogramma con il numero di bin specificato
hist_compton=TH1F("compton", "Spettro Compton 19 Marzo", num_bin, 0, 8192)
i=0
# Riempie i bin con le altezze specificate
for i in range(len(compton)):
	hist_compton.Fill(compton[i])


for i in range(len(compton2)):
	hist_compton.Fill(compton2[i])

	
# Adattamento della funzione ai dati
hist_compton.Fit("fit_function_compton", "ILR")
hist_compton.GetXaxis().SetTitle("ch")
hist_compton.GetYaxis().SetTitle("Eventi")

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

num_bins = hist_compton.GetNbinsX()
num_entries = hist_compton.GetEntries()
ndof = fit_function_compton.GetNDF()
chi_square = fit_function_compton.GetChisquare()
# Aggiungi una legenda al plot
legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
legend_compton.AddEntry("","Data di acquisizione : 19 Marzo, #theta=19 deg", "")
legend_compton.AddEntry("","Durata acquisizione: 52min", "")
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_compton.AddEntry(hist_compton, f"Histogram Entries: {num_entries:.0f}", "l")
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_compton.AddEntry(fit_function_compton, f"Fit function: gauss_{{1}}+gauss_{{2}}+parabola", "l")
# Aggiungi il numero di bin alla legenda
legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_compton.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_compton.AddEntry("", f"#mu_{{1}} = {p1_compton:.2f} #pm {p1_err_compton:.2f}, #sigma_{{1}} = {abs(p2_compton):.2f} #pm {p2_err_compton:.2f} ", "")
legend_compton.AddEntry("", f"#mu_{{2}} = {p4_compton:.2f} #pm {p4_err_compton:.2f}, #sigma_{{2}} = {abs(p5_compton):.2f} #pm {p5_err_compton:.2f} ", "")
legend_compton.AddEntry("", f"#chi^{{2}}/ndof = {chi_square:.0f}/{ndof:.0f} ", "")
# Disegna la legenda
legend_compton.Draw()

c_compton.Draw()


# Crea il canvas per il grafico dei residui
canvas = TCanvas("canvas", "Residual Plot", 800, 600)


# Crea un grafico per i residui
graph_residuals = TGraphErrors()

# Calcola i residui
point_index = 0
for i in range(1, hist_compton.GetNbinsX() + 1):
    x = hist_compton.GetBinCenter(i)
    if x < 4600 or x > 6200:
        continue
    y_data = hist_compton.GetBinContent(i)
    y_fit = fit_function_compton.Eval(x)
    residual = y_data - y_fit
    x_error = hist_compton.GetBinWidth(i) / (12 ** 0.5)
    y_error = (y_data ** 0.5)  # Errore poissoniano
    graph_residuals.SetPoint(point_index, x, residual)
    graph_residuals.SetPointError(point_index, x_error, y_error)
    point_index += 1

# Disegna il grafico dei residui
graph_residuals.Draw("AP")

# Disegna una linea a y=0
line = TLine(4600, 0, 6200, 0)
line.SetLineColor(kRed)
line.Draw()

# Aggiungi titoli e legenda
graph_residuals.SetTitle("Residual Plot")
graph_residuals.GetXaxis().SetTitle("Energy")
graph_residuals.GetYaxis().SetTitle("Residual")
graph_residuals.SetMarkerStyle(20)
line.SetLineStyle(2)

# Visualizza il canvas
canvas.Draw()


print("E finali in canali")
print(p1_compton, p1_err_compton)
print(p4_compton, p4_err_compton)

input()
#p1_compton += 20
#p4_compton += 20
"""
#*************************************************************************************************************************************************************************************

# FIT COMPTON PER SPETTRO IN FORMATO NON ESADECIMALE

#*************************************************************************************************************************************************************************************


# fit function double gauss+polynomial per 28 febbraio
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4000, 6600)
fit_function_compton.SetParameters(838.098, 5200.82, 221.044, 758.551, 5816.09, 206.753,  9130.62 , -3.0042, 0.000248927) 

compton=plb.loadtxt('02-28/compton/compton_73deg_28feb.dat', unpack=True)
compton2=plb.loadtxt('03-05/compton/compton_15deg_35cm.dat', unpack=True)
compton3=plb.loadtxt('03-06/compton/compton_25deg_35cm.dat', unpack=True)
p1_compton, p1_err_compton, p4_compton, p4_err_compton=compton_fit2(compton, fit_function_compton, 16)

p1_err_compton=np.sqrt( p1_err_compton**2+(dCompton1*p1_compton)**2)
p4_err_compton=np.sqrt( p4_err_compton**2+(dCompton2*p4_compton)**2)

print("E finali in canali")
print(p1_compton, p1_err_compton)
print(p4_compton, p4_err_compton)



"""


#*************************************************************************************************************************************************************************************

# MASSA ELETTRONE

#***************************************************************************************************************************************************************************************

m, q, c, b, a=np.loadtxt('dati/pars_cal_28e29febbraio_iniziofinemedia.txt', unpack=True)
dm, dq, dc, db, da=np.loadtxt('dati/err_pars_cal_28e29febbraio_iniziofinemedia.txt', unpack=True)

"""
m1, q1, c1, b1, a1=np.loadtxt('dati/pars_cal_28e29marzo_inizio.txt', unpack=True)
dm1, dq1, dc1, db1, da1=np.loadtxt('dati/err_pars_cal_28e29marzo_inizio.txt', unpack=True)
m2, q2, c2, b2, a2=np.loadtxt('dati/pars_cal_28e29marzo_fine.txt', unpack=True)
dm2, dq2, dc2, db2, da2=np.loadtxt('dati/err_pars_cal_28e29marzo_fine.txt', unpack=True)

def err_medio(m1, err_m1, m2, err_m2):
    # Calcola l'errore sulla media
    err_media = np.sqrt((err_m1**2 + err_m2**2) / 4)

    
    return err_media

m=(m1+m2)/2
q=(q1+q2)/2
c=(c1+c2)/2
b=(b1+b2)/2
a=(a1+a2)/2

dm=np.sqrt( ((m1-m2)/2)**2+err_medio(m1, dm1, m2, dm2)**2)
dq=np.sqrt( ((q1-q2)/2)**2+err_medio(q1, dq1, q2, dq2)**2)
dc=np.sqrt( ((c1-c2)/2)**2+err_medio(c1, dc1, c2, dc2)**2)
db=np.sqrt( ((b1-b2)/2)**2+err_medio(b1, db1, b2, db2)**2)
da=np.sqrt( ((a1-a2)/2)**2+err_medio(a1, da1, a2, da2)**2)
"""


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

cov_linear=np.loadtxt('dati/cov_lineare_28e29febbraio_iniziofinemedia.txt', unpack=True)
cov_parabola=np.loadtxt('dati/cov_parabola_28e29febbraio_iniziofinemedia.txt', unpack=True)
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

"""
#conversione E "tabulate" sorgente
E_in_a_lin=conversionelineare(media_ch_tabA, q, m)
E_in_b_lin=conversionelineare(media_ch_tabB, q, m)
E_in_a_err_lin=E_lin_err(m, q, dm, dq, media_ch_tabA, err_ch_tabA, cov_mq)
E_in_b_err_lin=E_lin_err(m, q, dm, dq, media_ch_tabB, err_ch_tabB, cov_mq)

E_in_a_pol=conversionepolinomiale(media_ch_tabA, a, b, c)
E_in_b_pol=conversionepolinomiale(media_ch_tabB, a, b, c)
E_in_a_err_pol=E_pol_err(media_ch_tabA, err_ch_tabA, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)
E_in_b_err_pol=E_pol_err(media_ch_tabB, err_ch_tabB, a, b, c, da, db, dc, cov_ab, cov_ac, cov_bc)
"""


print('\n')
print("E finali lin, err in perc")
print(E_a_lin, dE_a_lin, (dE_a_lin/E_a_lin)*100)
print(E_b_lin, dE_b_lin, (dE_b_lin/E_b_lin)*100)
print("E finali pol, err in perc")
print(E_a_pol, dE_a_pol, (dE_a_pol/E_a_pol)*100 )
print(E_b_pol, dE_b_pol, (dE_b_pol/E_b_pol)*100)
print('\n')


"""
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

#salvataggio E iniziali tabulate
E_in_lin=np.array([E_in_a_lin, E_in_b_lin])
dE_in_lin=np.array([E_in_a_err_lin, E_in_b_err_lin])
E_in_pol=np.array([E_in_a_pol, E_in_b_pol])
dE_in_pol=np.array([E_in_a_err_pol, E_in_b_err_pol])

np.savetxt('dati/E_iniziali_lin_21marzo_iniziofinemedia.txt', E_in_lin)
np.savetxt('dati/E_iniziali_lin_err_21marzo_iniziofinemedia.txt', dE_in_lin)
np.savetxt('dati/E_iniziali_pol_21marzo_iniziofinemedia.txt', E_in_pol )
np.savetxt('dati/E_iniziali_pol_err_21marzo_iniziofinemedia.txt', dE_in_pol )

#******************************************************************
"""



#salvataggio E finali E' scatterate
E_fin_lin=np.array([E_a_lin, E_b_lin])
dE_fin_lin=np.array([dE_a_lin, dE_b_lin])
E_fin_pol=np.array([E_a_pol, E_b_pol])
dE_fin_pol=np.array([dE_a_pol, dE_b_pol])

np.savetxt('dati/E_finali_lin_28e29febbraio_iniziofinemedia.txt', E_fin_lin)
np.savetxt('dati/E_finali_lin_err_28e29febbraio_iniziofinemedia.txt', dE_fin_lin)
np.savetxt('dati/E_finali_pol_28e29febbraio_iniziofinemedia.txt', E_fin_pol)
np.savetxt('dati/E_finali_pol_err_28e29febbraio_iniziofinemedia.txt', dE_fin_pol)

def m_E(E, E_in, theta):
	return (E*E_in*(1-np.cos(theta)))/(E_in-E)

	
#senza l'errore su E iniziale, si usa E tabulato	
def m_E_err(E, E_in, E_err, theta,  theta_err):
    
	d_m_E_dE = (E_in * (1 - np.cos(theta))) / (E_in - E) + (E * E_in * (1 - np.cos(theta))) / ((E_in - E) ** 2)
	d_m_E_dtheta = (E*E_in*np.sin(theta))/(E-E_in)
	m_E_err = np.sqrt((d_m_E_dE * E_err) ** 2 + (d_m_E_dtheta * theta_err) ** 2)

	return m_E_err


angolo=93+20-92.9536862495961

m_e1a_lin=m_E(E_a_lin, 1.173, angolo*np.pi/180)
m_e1b_lin=m_E(E_b_lin, 1.333, angolo*np.pi/180)
dm_e1a_lin=m_E_err(E_a_lin, 1.173,  dE_a_lin, angolo*np.pi/180, np.pi/(180*np.sqrt(1)))
dm_e1b_lin=m_E_err(E_b_lin, 1.333,  dE_b_lin, angolo*np.pi/180, np.pi/(180*np.sqrt(1)))

m_e1a_pol=m_E(E_a_pol, 1.173, angolo*np.pi/180)
m_e1b_pol=m_E(E_b_pol, 1.333, angolo*np.pi/180)
dm_e1a_pol=m_E_err(E_a_pol, 1.173,  dE_a_pol, angolo*np.pi/180, np.pi/(180*np.sqrt(1)))
dm_e1b_pol=m_E_err(E_b_pol, 1.333,  dE_b_pol, angolo*np.pi/180, np.pi/(180*np.sqrt(1)))



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




masse = np.array([
    ['#conversione_lineare', '#primo_picco', '#secondo_picco'],
    [m_e1a_lin*1000, dm_e1a_lin*1000, ''],
    [m_e1b_lin*1000, dm_e1b_lin*1000, ''],
    ['#conversione_quadratica', '#primo_picco', '#secondo_picco'],
    [m_e1a_pol*1000, dm_e1a_pol*1000, ''],
    [m_e1b_pol*1000, dm_e1b_pol*1000, '']
], dtype=object)

# Save the array to a text file
#np.savetxt('masse elettroni/masse_elettroni_28e29febbraio_mediate.txt', masse, fmt='%s')



