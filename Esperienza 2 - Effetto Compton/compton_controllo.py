import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from convertitore import convertitore

#********************************************

# funzioni di fit

#********************************************


# fit function double gauss+gauss bkg
fit_function_doublegaussgaussbkg = TF1("fit_function_doublegaussgaussbkg", "gaus(0)+gaus(3)+gaus(6)")

# fit function double gauss+expo
fit_function_doublegaussexpo = TF1("fit_function_doublegaussexpo", "gaus(0)+gaus(3)+expo(6)")

# fit function double gauss+polynomial
fit_function_doublegausspol = TF1("fit_function_doublegausspol", "gaus(0)+gaus(3)+pol2(6)")

# fit function double gauss+retta
fit_function_doublegaussretta = TF1("fit_function_doublegaussretta", "gaus(0)+gaus(3)+pol1(6)")




#***************************************************

# DATI VARI SPETTRI

#***************************************************
"""
compton1=plb.loadtxt("02-27/compton/compton_63deg_27feb.dat")
compton2=plb.loadtxt("02-28/compton/compton_73deg_28feb.dat")
compton3=plb.loadtxt("02-29/compton/compton_10deg.dat")
compton4=plb.loadtxt("03-05/compton/compton_15deg_35cm.dat")
compton5=plb.loadtxt("03-06/compton/compton_25deg_35cm.dat")
compton6=convertitore("03-07/compton/compton_18deg_35cm_esadecimale.dat")
"""

compton1=convertitore("03-13/compton/compton_113deg_35cm_esadecimale.dat")
compton2a=convertitore("03-19/compton/compton_78deg-1_esadecimale.dat")
compton2b=convertitore("03-19/compton/compton_78deg-2_esadecimale.dat")
compton3= convertitore("03-20/compton/compton_22deg_esadecimale.dat")
compton4a= convertitore("03-21/compton/compton_15deg_esadecimale.dat")
compton4b= convertitore("03-21/compton/compton_15deg_2_esadecimale.dat")


#**************************************************

# FIT COMPTON

#**************************************************

# fit function double gauss+polynomial per 13 marzo
fit_function_compton1= TF1("fit_function_compton1", "gaus(0)+gaus(3)+pol2(6)", 5050, 6710)
fit_function_compton1.SetParameters(710, 5584, 132, 725, 6308, 143, 13461, -3.85, 0.0002762 ) 
fit_function_compton1.SetLineColor(kBlue)
# fit function double gauss+polynomial per 19 marzo
fit_function_compton2= TF1("fit_function_compton2", "gaus(0)+gaus(3)+pol2(6)", 4600, 6000)
fit_function_compton2.SetParameters(150, 5057.5, 185.392,  142.71,  5628.39, 205.4,    1895.42,   -0.628374,  5.25085e-05 )
fit_function_compton2.SetLineColor(kRed)
# fit function double gauss+polynomial per 20e21 marzo
fit_function_compton3= TF1("fit_function_compton3", "gaus(0)+gaus(3)+pol2(6)", 4000, 6000)
fit_function_compton3.SetParameters(1818, 4914, 210, 1467, 5496, 189, 5783, -1.19, 4.2e-05 )
fit_function_compton3.SetLineColor(kBlack)

# fit function double gauss+polynomial per 21 marzo
fit_function_compton4= TF1("fit_function_compton4", "gaus(0)+gaus(3)+pol2(6)", 4200, 6400)
fit_function_compton4.SetParameters(180, 4914, 140, 211, 5496, 189, 328, 0.04, -1.37e-05)
fit_function_compton4.SetLineColor(kMagenta)


c_compton = TCanvas("compton", "compton")

hist_compton1=TH1F("compton1", "compton1", 400, 0, 8192)
hist_compton2=TH1F("compton2", "compton2", 250, 0, 8192)
hist_compton3=TH1F("compton3", "compton3", 400, 0, 8192)
hist_compton4=TH1F("compton4", "compton4", 250, 0, 8192)


i=0
for i in range(len(compton1)):
	hist_compton1.Fill(compton1[i])
	

i=0
for i in range(len(compton2a)):
	hist_compton2.Fill(compton2a[i])
	i=0
for i in range(len(compton2b)):
	hist_compton2.Fill(compton2b[i])


i=0
for i in range(len(compton3)):
	hist_compton3.Fill(compton3[i])

i=0
for i in range(len(compton4a)):
	hist_compton4.Fill(compton4a[i])
i=0
for i in range(len(compton4b)):
	hist_compton4.Fill(compton4b[i])
	



hist_compton1.Fit("fit_function_compton1", "ILR")
hist_compton2.Fit("fit_function_compton2", "ILR")
hist_compton3.Fit("fit_function_compton3", "ILR")
hist_compton4.Fit("fit_function_compton4", "ILR")

hist_compton3.Draw()
fit_function_compton3.Draw("same")
hist_compton4.Draw("same")
fit_function_compton4.Draw("same")
hist_compton1.Draw("same")
fit_function_compton1.Draw("same")
hist_compton2.Draw("same")
fit_function_compton2.Draw("same")



c_compton.Draw()

input()
