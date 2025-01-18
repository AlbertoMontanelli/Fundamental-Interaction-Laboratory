import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
from ROOT import *


fit_function_Am = TF1("fit_function_Am", "gaus(0)+pol2(3)", 152, 464)
fit_function_Am.SetParameters(379824, 311, 21, -10197, 123, -0.21)


def fit_cobalto2(cobalto):
	c_Co=TCanvas("cobalto", "cobalto")
	fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
	fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000) 
	
	#decido i bin 
	num_bin = len(cobalto)
	hist_cobalto=TH1F("cobalto", "cobalto", num_bin, 0, num_bin)
	i=0
	# Riempie i bin con le altezze specificate
	for i in range(num_bin):
    		hist_cobalto.SetBinContent(i+1, cobalto[i])
    		
	hist_cobalto.Rebin(8)
	
	# Adattamento della funzione ai dati
	hist_cobalto.Fit("fit_function_Co", "ILR")

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
	
	hist_cobalto.Draw()
	fit_function_Co.Draw("same")
	c_Co.Draw()
	input()	

	return p1_Co, p1_err_Co, p4_Co, p4_err_Co
	

def fit_sodio2(sodio):
	c_Na=TCanvas("sodio", "sodio")
	fit_function_Na = TF1("fit_function_Na", "gaus(0)+pol2(3)", 2000, 3000)
	fit_function_Na.SetParameters(1531, 2633, 89,  1026, -0.620508, 0.000104099)
	fit_function_Na2 = TF1("fit_function_Na2", "gaus(0)+pol1(3)", 5600, 6800)
	fit_function_Na2.SetParameters(126, 6362, 134,  83.1, -0.007401)
	
	#decido i bin 
	num_bin = len(sodio)
	hist_sodio=TH1F("sodio", "sodio", num_bin, 0, num_bin)
	i=0
	# Riempie i bin con le altezze specificate
	for i in range(num_bin):
    		hist_sodio.SetBinContent(i+1, sodio[i])
    		
	hist_sodio.Rebin(8)

	# Adattamento della funzione ai dati
	hist_sodio.Fit("fit_function_Na", "ILR")
	hist_sodio.Fit("fit_function_Na2", "ILR")

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
	hist_sodio.Draw()
	fit_function_Na.Draw("same")
	fit_function_Na2.Draw("same")
	c_Na.Draw()
	input()
	return p1_Na, p1_err_Na, p1_Na_2, p1_err_Na_2
	
	
def fit_cesio2(cesio):
	c_Cs=TCanvas("cesio", "cesio")
	fit_function_Cs = TF1("fit_function_Cs", "gaus(0)+pol2(3)", 2600, 4016)
	fit_function_Cs.SetParameters(32373, 3450, -104, 3567, -0.18, -0.00017086)
	#decido i bin 
	num_bin = len(cesio)
	# Crea un istogramma con il numero di bin specificato
	hist_cesio=TH1F("cesio", "cesio", num_bin, 0, num_bin)
	i=0
	for i in range(num_bin):
    		hist_cesio.SetBinContent(i+1, cesio[i])
    		
	hist_cesio.Rebin(8)
    	
    	# Adattamento della funzione ai dati
	hist_cesio.Fit("fit_function_Cs", "ILR")
	#parametri fit
	p0_Cs = fit_function_Cs.GetParameter(0)
	p0_err_Cs = fit_function_Cs.GetParError(0)
	p1_Cs = fit_function_Cs.GetParameter(1)
	p1_err_Cs = fit_function_Cs.GetParError(1)
	p2_Cs = fit_function_Cs.GetParameter(2)
	p2_err_Cs = fit_function_Cs.GetParError(2)
	p3_Cs = fit_function_Cs.GetParameter(3)
	p3_err_Cs = fit_function_Cs.GetParError(3)
	p4_Cs = fit_function_Cs.GetParameter(4)
	p4_err_Cs = fit_function_Cs.GetParError(4)
	p5_Cs = fit_function_Cs.GetParameter(5)
	p5_err_Cs = fit_function_Cs.GetParError(5)
	hist_cesio.Draw()
	fit_function_Cs.Draw("same")
	c_Cs.Draw()
	input()	

	return p1_Cs, p1_err_Cs
