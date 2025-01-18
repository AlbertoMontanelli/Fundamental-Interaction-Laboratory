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


#***************************************************

# DATI VARIE SORGENTI

#***************************************************

cobalto1= plb.loadtxt('02-27/calibrazione/cobalto.dat',unpack=True)
cobalto2= plb.loadtxt('02-28/calibrazione/cobalto.dat',unpack=True)
cobalto3= plb.loadtxt('02-29/calibrazione/cobalto.dat',unpack=True)
cobalto4= plb.loadtxt('03-05/calibrazione/cobalto.dat',unpack=True)
cobalto5= plb.loadtxt('03-06/calibrazione/cobalto.dat',unpack=True)
cobalto6=convertitore("03-07/calibrazione/cobalto_esadecimale.dat")
cobalto7=convertitore("03-12/calibrazione/inizio/cobalto_esadecimale.dat")
cobalto8=convertitore("03-12/calibrazione/fine/cobalto_esadecimale.dat")
cobalto9=convertitore("03-13/calibrazione/inizio/cobalto_esadecimale.dat")
cobalto10=convertitore("03-13/calibrazione/fine/cobalto_esadecimale.dat")
cobalto11=convertitore("03-19/calibrazione/inizio/cobalto_inizio_esadecimale.dat")
cobalto12=convertitore("03-19/calibrazione/fine/cobalto_fine_esadecimale.dat")


sodio1 = plb.loadtxt('02-27/calibrazione/sodio.dat', unpack=True)
sodio2 = plb.loadtxt('02-28/calibrazione/sodio.dat', unpack=True)
sodio3 = plb.loadtxt('02-29/calibrazione/sodio.dat', unpack=True)
sodio4 = plb.loadtxt('03-05/calibrazione/sodio.dat', unpack=True)
sodio5 = plb.loadtxt('03-06/calibrazione/sodio.dat', unpack=True)
sodio6 = convertitore("03-07/calibrazione/sodio_esadecimale.dat")
sodio7 = convertitore("03-12/calibrazione/inizio/sodio_esadecimale.dat")
sodio8 = convertitore("03-12/calibrazione/fine/sodio_esadecimale.dat")
sodio9 = convertitore("03-13/calibrazione/inizio/sodio_esadecimale.dat")
sodio10 = convertitore("03-13/calibrazione/fine/sodio_esadecimale.dat")
sodio11=convertitore("03-19/calibrazione/inizio/sodio_inizio_esadecimale.dat")
sodio12=convertitore("03-19/calibrazione/fine/sodio_fine_esadecimale.dat")

cesio1 = plb.loadtxt('02-27/calibrazione/cesio.dat', unpack=True)
cesio2 = plb.loadtxt('02-28/calibrazione/cesio.dat', unpack=True)
cesio3 = plb.loadtxt('02-29/calibrazione/cesio.dat', unpack=True)
cesio4 = plb.loadtxt('03-05/calibrazione/cesio.dat', unpack=True)
cesio5 = plb.loadtxt('03-06/calibrazione/cesio.dat', unpack=True)
cesio6 = convertitore("03-07/calibrazione/cesio_esadecimale.dat")
cesio7 = convertitore("03-12/calibrazione/inizio/cesio_esadecimale.dat")
cesio8 = convertitore("03-12/calibrazione/fine/cesio_esadecimale.dat")
cesio9 = convertitore("03-13/calibrazione/inizio/cesio_esadecimale.dat")
cesio10 = convertitore("03-13/calibrazione/fine/cesio_esadecimale.dat")
cesio11=convertitore("03-19/calibrazione/inizio/cesio_inizio_esadecimale.dat")
cesio12=convertitore("03-19/calibrazione/fine/cesio_fine_esadecimale.dat")


#*********************************************************************

#FIT SORGENTI

#*************************************************************************


#************************************************************************

# COBALTO

#***************************************************************************

c_cobalto9 = TCanvas("cobalto9", "cobalto9")

fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000) 
#decido i bin 
num_bin = 500
# Crea un istogramma con il numero di bin specificato
hist_cobalto9=TH1F("cobalto9", "cobalto9", num_bin, 0, 8192)
for i in range(len(cobalto9)):
    hist_cobalto9.Fill(cobalto9[i])
    
# Adattamento della funzione ai dati
hist_cobalto9.Fit("fit_function_Co", "ILR")


#plot
hist_cobalto9.Draw()
fit_function_Co.Draw("same")
c_cobalto9.Draw()

input()


#**********************************************************************************

c_cobalto10 = TCanvas("cobalto10", "cobalto10")

fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000) 
#decido i bin 
num_bin = 500
# Crea un istogramma con il numero di bin specificato
hist_cobalto10=TH1F("cobalto10", "cobalto10", num_bin, 0, 8192)
for i in range(len(cobalto10)):
    hist_cobalto10.Fill(cobalto10[i])
    
# Adattamento della funzione ai dati
hist_cobalto10.Fit("fit_function_Co", "ILR")


#plot
hist_cobalto10.Draw()
fit_function_Co.Draw("same")
c_cobalto10.Draw()




input()




#**************************************************************************

#SODIO

#****************************************************************************
c_sodio9 = TCanvas("sodio9", "sodio9")

fit_function_Na = TF1("fit_function_Na", "gaus(0)+pol2(3)", 2000, 3000)
fit_function_Na.SetParameters(1531, 2633, 89,  1026, -0.620508, 0.000104099)
fit_function_Na2 = TF1("fit_function_Na2", "gaus(0)+pol1(3)", 5600, 6800)
fit_function_Na2.SetParameters(126, 6362, 134,  83.1, -0.007401)
#decido i bin 
num_bin = 500
# Crea un istogramma con il numero di bin specificato
hist_sodio9=TH1F("sodio9", "sodio9", num_bin, 0, 8192)
for i in range(len(sodio9)):
    hist_sodio9.Fill(sodio9[i])

# Adattamento della funzione ai dati
hist_sodio9.Fit("fit_function_Na", "ILR")
hist_sodio9.Fit("fit_function_Na2", "ILR")


#plot
hist_sodio9.Draw()
fit_function_Na.Draw("same")
fit_function_Na2.Draw("same")
c_sodio9.Draw()

input()


#***************************************************************************
c_sodio10 = TCanvas("sodio10", "sodio10")

fit_function_Na = TF1("fit_function_Na", "gaus(0)+pol2(3)", 2000, 3000)
fit_function_Na.SetParameters(1531, 2633, 89,  1026, -0.620508, 0.000104099)
fit_function_Na2 = TF1("fit_function_Na2", "gaus(0)+pol1(3)", 5600, 6800)
fit_function_Na2.SetParameters(126, 6362, 134,  83.1, -0.007401)
#decido i bin 
num_bin = 500
# Crea un istogramma con il numero di bin specificato
hist_sodio10=TH1F("sodio10", "sodio10", num_bin, 0, 8192)
for i in range(len(sodio10)):
    hist_sodio10.Fill(sodio10[i])

# Adattamento della funzione ai dati
hist_sodio10.Fit("fit_function_Na", "ILR")
hist_sodio10.Fit("fit_function_Na2", "ILR")


#plot
hist_sodio10.Draw()
fit_function_Na.Draw("same")
fit_function_Na2.Draw("same")
c_sodio10.Draw()

input()


#**************************************************************************

# CESIO

#***************************************************************************
fit_function_Cs = TF1("fit_function_Cs", "gaus(0)+pol2(3)", 2600, 4016)
fit_function_Cs.SetParameters(32373, 3450, -104, 3567, -0.18, -0.00017086)

# Creo il canvas
c_cesio9 = TCanvas("cesio9", "cesio9")

num_bin = 500

# Crea un istogramma con il numero di bin specificato
hist_cesio9 = TH1F("cesio9", "cesio9", num_bin, 0, 8192)
for i in range(len(cesio9)):
    hist_cesio9.Fill(cesio9[i])
    	
# Adattamento della funzione ai dati
hist_cesio9.Fit("fit_function_Cs", "ILR")

# Plot
hist_cesio9.Draw()
fit_function_Cs.Draw("same")
c_cesio9.Draw()

input()


#**************************************************************************

fit_function_Cs = TF1("fit_function_Cs", "gaus(0)+pol2(3)", 2600, 4016)
fit_function_Cs.SetParameters(32373, 3450, -104, 3567, -0.18, -0.00017086)

# Creo il canvas
c_cesio10 = TCanvas("cesio10", "cesio10")

num_bin = 500

# Crea un istogramma con il numero di bin specificato
hist_cesio10 = TH1F("cesio10", "cesio10", num_bin, 0, 8192)
for i in range(len(cesio10)):
    hist_cesio10.Fill(cesio10[i])
    	
# Adattamento della funzione ai dati
hist_cesio10.Fit("fit_function_Cs", "ILR")

# Plot
hist_cesio10.Draw()
fit_function_Cs.Draw("same")
c_cesio10.Draw()

input()


