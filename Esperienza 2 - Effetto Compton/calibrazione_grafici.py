import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *


#********************************************

# funzioni di fit

#********************************************


# fit function gauss+gauss bkg
fit_function_gaussgaussbkg = TF1("fit_function_gaussgaussbkg", "gaus(0)+gaus(3)", 5500, 6200)

# fit function gauss+pol
fit_function_gausspol = TF1("fit_function_gausspol", "gaus(0)+pol2(3)", 5500, 6200)

# fit function gauss+expo
fit_function_gaussexpo = TF1("fit_function_gaussexpo", "gaus(0)+expo(3)", 5500, 6200)

# fit function double gauss+gauss bkg
fit_function_doublegaussgaussbkg = TF1("fit_function_doublegaussgaussbkg", "gaus(0)+gaus(3)+gaus(6)", 5500, 6200)

# fit function double gauss+expo
fit_function_doublegaussexpo = TF1("fit_function_doublegaussexpo", "gaus(0)+gaus(3)+expo(6)", 4600, 7500)

# fit function double gauss+polynomial
fit_function_doublegausspol = TF1("fit_function_doublegausspol", "gaus(0)+gaus(3)+pol2(6)", 4600, 7500)

# fit function double gauss+retta
fit_function_doublegaussretta = TF1("fit_function_doublegaussretta", "gaus(0)+gaus(3)+pol1(6)", 5000, 7500)


#*********************************************

# COBALTO

#*********************************************


#6 fit function double gauss+gauss bkg
fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
fit_function_Co.SetParameters(2000, 5922, -131, 1600, 6500, -131, 1400, 700, 500)  # (ampiezza, media, deviazione standard, ampiezza, slope, offset)

cobalto = plb.loadtxt('27 febbraio/calibrazione/cobalto_14e40.dat',unpack=True)

#creo il canvas
c_cobalto = TCanvas("cobalto", "cobalto")

#decido i bin 
num_bin = len(cobalto)
# Crea un istogramma con il numero di bin specificato
hist_cobalto=TH1F("cobalto", "cobalto", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
    hist_cobalto.SetBinContent(i, cobalto[i])

#rebinning
hist_cobalto.Rebin(8)
entries=0.

#entries
for i in range(len(cobalto)):
	entries=entries+cobalto[i]


# Adattamento della funzione ai dati
hist_cobalto.Fit("fit_function_Co", "R")

#plot
hist_cobalto.Draw()
fit_function_Co.Draw("same")

#chi quadro/gradi di libertà
ndof = fit_function_Co.GetNDF()
chi_square = fit_function_Co.GetChisquare()
#numero bin
num_bins = hist_cobalto.GetNbinsX()
#parametri fit
p0_Co = fit_function_Co.GetParameter(0)
p0_err_Co = fit_function_Co.GetParError(0)
p1_Co = fit_function_Co.GetParameter(1)
p1_err_Co = fit_function_Co.GetParError(1)
p2_Co = fit_function_Co.GetParameter(2)
p2_err_Co = fit_function_Co.GetParError(2)
p3_Co = fit_function_Co.GetParameter(3)
p3_err_Co = fit_function_Co.GetParError(3)
p4_Co = fit_function_Co.GetParameter(4)
p4_err_Co = fit_function_Co.GetParError(4)
p5_Co = fit_function_Co.GetParameter(5)
p5_err_Co = fit_function_Co.GetParError(5)
p6_Co = fit_function_Co.GetParameter(6)
p6_err_Co = fit_function_Co.GetParError(6)
p7_Co = fit_function_Co.GetParameter(7)
p7_err_Co = fit_function_Co.GetParError(7)
p8_Co = fit_function_Co.GetParameter(8)
p8_err_Co = fit_function_Co.GetParError(8)

# Aggiungi una legenda al plot
legend_cobalto = TLegend(0.7, 0.6, 0.98, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_cobalto.AddEntry(hist_cobalto, f"Histogram Entries: {entries:.0f}", "l")
# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend_cobalto.AddEntry(fit_function_Co, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+A_{2}e^{-#frac{(x-#mu_{2})^{2}}{2#sigma_{2}^{2}}}+A_{bkg}e^{-#frac{(x-#mu_{bkg})^{2}}{2#sigma_{bkg}^{2}}}", "l")
# Aggiungi il numero di bin alla legenda
legend_cobalto.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_cobalto.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_cobalto.AddEntry("", f"#mu_{{1}} = {p1_Co:.2f} #pm {p1_err_Co:.2f}, #sigma_{{1}} = {abs(p2_Co):.2f} #pm {p2_err_Co:.2f} ", "")
legend_cobalto.AddEntry("", f"#mu_{{2}} = {p4_Co:.2f} #pm {p4_err_Co:.2f}, #sigma_{{2}} = {abs(p5_Co):.2f} #pm {p5_err_Co:.2f} ", "")
legend_cobalto.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
# Disegna la legenda
legend_cobalto.Draw()

c_cobalto.Draw()


#*********************************************

# SODIO

#*********************************************

#1 fit function gauss+polynomial
fit_function_Na = TF1("fit_function_Na", "gaus(0)+pol2(3)", 2000, 3000)
fit_function_Na.SetParameters(1531, 2633, 89,  1026, -0.620508, 0.000104099) 
fit_function_Na2 = TF1("fit_function_Na2", "gaus(0)+pol1(3)", 6000, 6800)
fit_function_Na2.SetParameters(126, 6362, 134,  83.1, -0.007401) 
sodio = plb.loadtxt('27 febbraio/calibrazione/sodio_15e30.dat',unpack=True)

#creo il canvas
c_sodio = TCanvas("sodio", "sodio")

#decido i bin 
num_bin = len(sodio)
# Crea un istogramma con il numero di bin specificato
hist_sodio=TH1F("sodio", "sodio", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
    hist_sodio.SetBinContent(i, sodio[i])

#rebinning
hist_sodio.Rebin(8)
entries=0.

#entries
for i in range(len(sodio)):
	entries=entries+sodio[i]


# Adattamento della funzione ai dati
hist_sodio.Fit("fit_function_Na", "R")
hist_sodio.Fit("fit_function_Na2", "R")


#plot
hist_sodio.Draw()
fit_function_Na.Draw("same")
fit_function_Na2.Draw("same")

#chi quadro/gradi di libertà
ndof = fit_function_Na.GetNDF()
chi_square = fit_function_Na.GetChisquare()
#numero bin
num_bins = hist_sodio.GetNbinsX()
#parametri fit
p0_Na = fit_function_Na.GetParameter(0)
p0_err_Na = fit_function_Na.GetParError(0)
p1_Na = fit_function_Na.GetParameter(1)
p1_err_Na = fit_function_Na.GetParError(1)
p2_Na = fit_function_Na.GetParameter(2)
p2_err_Na = fit_function_Na.GetParError(2)
p3_Na = fit_function_Na.GetParameter(3)
p3_err_Na = fit_function_Na.GetParError(3)
p4_Na = fit_function_Na.GetParameter(4)
p4_err_Na = fit_function_Na.GetParError(4)
p5_Na = fit_function_Na.GetParameter(5)
p5_err_Na = fit_function_Na.GetParError(5)

p0_Na_2 = fit_function_Na2.GetParameter(0)
p0_err_Na_2 = fit_function_Na2.GetParError(0)
p1_Na_2 = fit_function_Na2.GetParameter(1)
p1_err_Na_2 = fit_function_Na2.GetParError(1)
p2_Na_2 = fit_function_Na2.GetParameter(2)
p2_err_Na_2 = fit_function_Na2.GetParError(2)
p3_Na_2 = fit_function_Na2.GetParameter(3)
p3_err_Na_2 = fit_function_Na2.GetParError(3)
p4_Na_2 = fit_function_Na2.GetParameter(4)
p4_err_Na_2 = fit_function_Na2.GetParError(4)
p5_Na_2 = fit_function_Na2.GetParameter(5)
p5_err_Na_2 = fit_function_Na2.GetParError(5)

# Aggiungi una legenda al plot
legend_sodio = TLegend(0.7, 0.6, 0.98, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_sodio.AddEntry(hist_sodio, f"Histogram Entries: {entries:.0f}", "l")
# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend_sodio.AddEntry(fit_function_Na, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+ax^{2}+bx+c", "l")
legend_sodio.AddEntry(fit_function_Na2, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+mx+q", "l")
# Aggiungi il numero di bin alla legenda
legend_sodio.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_sodio.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_sodio.AddEntry("", f"#mu_{{1}} = {p1_Na:.2f} #pm {p1_err_Na:.2f}, #sigma_{{1}} = {abs(p2_Na):.2f} #pm {p2_err_Na:.2f} ", "")
legend_sodio.AddEntry("", f"#mu_{{2}} = {p1_Na_2:.2f} #pm {p1_err_Na_2:.2f}, #sigma_{{1}} = {abs(p2_Na_2):.2f} #pm {p2_err_Na_2:.2f} ", "")
legend_sodio.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
# Disegna la legenda
legend_sodio.Draw()

c_sodio.Draw()



#*********************************************

# CESIO

#*********************************************

#1 fit function gauss+polynomial
fit_function_Ce = TF1("fit_function_Ce", "gaus(0)+pol2(3)", 2800, 4016)
fit_function_Ce.SetParameters(32373, 3450, -104, 3567, -0.18, -0.00017086) # (ampiezza, media, deviazione standard, c, b. a)

cesio = plb.loadtxt('27 febbraio/calibrazione/cesio_15e10.dat',unpack=True)

#creo il canvas
c_cesio = TCanvas("cesio", "cesio")

#decido i bin 
num_bin = len(cesio)
# Crea un istogramma con il numero di bin specificato
hist_cesio=TH1F("cesio", "cesio", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
    hist_cesio.SetBinContent(i, cesio[i])

#rebinning
hist_cesio.Rebin(8)
entries=0.

#entries
for i in range(len(cesio)):
	entries=entries+cesio[i]


# Adattamento della funzione ai dati
hist_cesio.Fit("fit_function_Ce", "R")



#plot
hist_cesio.Draw()
fit_function_Ce.Draw("same")

#chi quadro/gradi di libertà
ndof = fit_function_Ce.GetNDF()
chi_square = fit_function_Ce.GetChisquare()
#numero bin
num_bins = hist_cesio.GetNbinsX()
#parametri fit
p0_Ce = fit_function_Ce.GetParameter(0)
p0_err_Ce = fit_function_Ce.GetParError(0)
p1_Ce = fit_function_Ce.GetParameter(1)
p1_err_Ce = fit_function_Ce.GetParError(1)
p2_Ce = fit_function_Ce.GetParameter(2)
p2_err_Ce = fit_function_Ce.GetParError(2)
p3_Ce = fit_function_Ce.GetParameter(3)
p3_err_Ce = fit_function_Ce.GetParError(3)
p4_Ce = fit_function_Ce.GetParameter(4)
p4_err_Ce = fit_function_Ce.GetParError(4)
p5_Ce = fit_function_Ce.GetParameter(5)
p5_err_Ce = fit_function_Ce.GetParError(5)

# Aggiungi una legenda al plot
legend_cesio = TLegend(0.7, 0.6, 0.98, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_cesio.AddEntry(hist_cesio, f"Histogram Entries: {entries:.0f}", "l")
# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend_cesio.AddEntry(fit_function_Ce, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+ax^{2}+bx+c", "l")
# Aggiungi il numero di bin alla legenda
legend_cesio.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_cesio.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_cesio.AddEntry("", f"#mu_{{1}} = {p1_Ce:.2f} #pm {p1_err_Ce:.2f}, #sigma_{{1}} = {abs(p2_Ce):.2f} #pm {p2_err_Ce:.2f} ", "")
legend_cesio.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
# Disegna la legenda
legend_cesio.Draw()

c_cesio.Draw()


#*********************************************

# STRONZIO

#*********************************************

#1 fit function gauss+polynomial
fit_function_Sr = TF1("fit_function_Sr", "gaus(0)+pol2(3)", 240, 1334)
fit_function_Sr.SetParameters(4200, 300, 293, 6063, -6.13, 0.00170) # (ampiezza, media, deviazione standard)

stronzio = plb.loadtxt('27 febbraio/calibrazione/stronzio_15e55.dat',unpack=True)

#creo il canvas
c_stronzio = TCanvas("stronzio", "stronzio")

#decido i bin 
num_bin = len(stronzio)
# Crea un istogramma con il numero di bin specificato
hist_stronzio=TH1F("stronzio", "stronzio", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
    hist_stronzio.SetBinContent(i, stronzio[i])

#rebinning
hist_stronzio.Rebin(8)
entries=0.

#entries
for i in range(len(stronzio)):
	entries=entries+stronzio[i]

"""
# Adattamento della funzione ai dati
hist_stronzio.Fit("fit_function_Sr", "R")
"""


#plot
hist_stronzio.Draw()
"""
fit_function_Sr.Draw("same")

#chi quadro/gradi di libertà
ndof = fit_function_Sr.GetNDF()
chi_square = fit_function_Sr.GetChisquare()
#numero bin
num_bins = hist_stronzio.GetNbinsX()
#parametri fit
p0_Sr = fit_function_Sr.GetParameter(0)
p0_err_Sr = fit_function_Sr.GetParError(0)
p1_Sr = fit_function_Sr.GetParameter(1)
p1_err_Sr = fit_function_Sr.GetParError(1)
p2_Sr = fit_function_Sr.GetParameter(2)
p2_err_Sr = fit_function_Sr.GetParError(2)
p3_Sr = fit_function_Sr.GetParameter(3)
p3_err_Sr = fit_function_Sr.GetParError(3)
p4_Sr = fit_function_Sr.GetParameter(4)
p4_err_Sr = fit_function_Sr.GetParError(4)
p5_Sr = fit_function_Sr.GetParameter(5)
p5_err_Sr = fit_function_Sr.GetParError(5)


# Aggiungi una legenda al plot
legend_stronzio = TLegend(0.7, 0.6, 0.98, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_stronzio.AddEntry(hist_stronzio, f"Histogram Entries: {entries:.0f}", "l")
# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend_stronzio.AddEntry(fit_function_Sr, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+ax^{2}+bx+c", "l")
# Aggiungi il numero di bin alla legenda
legend_stronzio.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_stronzio.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_stronzio.AddEntry("", f"#mu_{{1}} = {p1_Sr:.2f} #pm {p1_err_Sr:.2f}, #sigma_{{1}} = {abs(p2_Sr):.2f} #pm {p2_err_Sr:.2f} ", "")
legend_stronzio.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
# Disegna la legenda
legend_stronzio.Draw()
"""
c_stronzio.Draw()


#*********************************************

# AMERICIO

#*********************************************

#1 fit function gauss+polynomial
fit_function_Am = TF1("fit_function_Am", "gaus(0)+pol2(3)", 152, 464)
fit_function_Am.SetParameters(379824, 311, 21, -10197, 123, -0.21) # (ampiezza, media, deviazione standard, c, b. a)

americio = plb.loadtxt('27 febbraio/calibrazione/americio_15e00.dat',unpack=True)

#creo il canvas
c_americio = TCanvas("americio", "americio")

#decido i bin 
num_bin = len(americio)
# Crea un istogramma con il numero di bin specificato
hist_americio=TH1F("americio", "americio", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
    hist_americio.SetBinContent(i, americio[i])

#rebinning
hist_americio.Rebin(8)
entries=0.

#entries
for i in range(len(americio)):
	entries=entries+americio[i]


# Adattamento della funzione ai dati
hist_americio.Fit("fit_function_Am", "R")



#plot
hist_americio.Draw()
fit_function_Am.Draw("same")

#chi quadro/gradi di libertà
ndof = fit_function_Am.GetNDF()
chi_square = fit_function_Am.GetChisquare()
#numero bin
num_bins = hist_americio.GetNbinsX()
#parametri fit
p0_Am = fit_function_Am.GetParameter(0)
p0_err_Am = fit_function_Am.GetParError(0)
p1_Am = fit_function_Am.GetParameter(1)
p1_err_Am = fit_function_Am.GetParError(1)
p2_Am = fit_function_Am.GetParameter(2)
p2_err_Am = fit_function_Am.GetParError(2)
p3_Am = fit_function_Am.GetParameter(3)
p3_err_Am = fit_function_Am.GetParError(3)
p4_Am = fit_function_Am.GetParameter(4)
p4_err_Am = fit_function_Am.GetParError(4)
p5_Am = fit_function_Am.GetParameter(5)
p5_err_Am = fit_function_Am.GetParError(5)


# Aggiungi una legenda al plot
legend_americio = TLegend(0.7, 0.6, 0.98, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_americio.AddEntry(hist_americio, f"Histogram Entries: {entries:.0f}", "l")

"""
# Ridimensiona la legenda per fare spazio alla forma matematica
legend_americio.SetX1(0.1)  # Modifica la coordinata X in basso a sinistra
legend_americio.SetY1(0.5)  # Modifica la coordinata Y in basso a sinistra
legend_americio.SetX2(0.78)  # Modifica la coordinata X in alto a destra
legend_americio.SetY2(0.98)  # Modifica la coordinata Y in alto a destra
"""

# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend_americio.AddEntry(fit_function_Am, "Fit Function: A_{1}e^{- #frac{(x-#mu_{1})^{2}}{2#sigma_{1}^{2}}}+ax^{2}+bx+c", "l")
# Aggiungi il numero di bin alla legenda
legend_americio.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la riga "Parameters:" alla legenda
legend_americio.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_americio.AddEntry("", f"#mu_{{1}} = {p1_Am:.2f} #pm {p1_err_Am:.2f}, #sigma_{{1}} = {abs(p2_Am):.2f} #pm {p2_err_Am:.2f} ", "")
legend_americio.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
# Disegna la legenda
legend_americio.Draw()

c_americio.Draw()

#**********************************************************************

# CONVERSIONE CANALI->ENERGIA

#**********************************************************************

ch=np.array([p1_Co, p4_Co, p1_Na_2, p1_Ce])
dch=np.array([p1_err_Co, p4_err_Co,p1_err_Na_2, p1_err_Ce])
E=np.array([1.173, 1.333, 1.275, 0.662])



#creo il canvas
c_cal = TCanvas("calibrazione", "calibrazione")

lineare =TGraphErrors(4, E, ch, 0, dch)
lineare.SetLineColor(kBlack)  # Imposta il colore dei punti

lineare.SetMarkerStyle(20)  # Set marker style
lineare.SetMarkerSize(1)    # Set marker size

# Imposta l'aspetto del grafico
lineare.SetFillStyle(0)  # Imposta stile di riempimento trasparente
lineare.SetFillColor(0)  # Imposta colore di riempimento trasparente



# Aggiungi le label agli assi
lineare.GetXaxis().SetTitle("E [MeV]")  
lineare.GetYaxis().SetTitle("ch")
lineare.Fit("pol2")
# Disegna solo i punti senza linee
lineare.Draw("AP")  # "AP" indica di disegnare i punti con barre di errore

"""
#parametri fit
c = pol2.GetParameter(0)
c_err = pol2.GetParError(0)
b = pol2.GetParameter(1)
b_err_Am = pol2.GetParError(1)
a = pol2.GetParameter(2)
a_err = pol2.GetParError(2)
"""


c_cal.Draw()


"""
plb.savetxt('channel.txt',ch)
plb.savetxt('err_channel.txt',dch)
"""

input()



