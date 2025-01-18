import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *

def energie_compton(cobalto_sorgente, num_bin):


 

	#creo il canvas
	c_cobalto_sorgente = TCanvas("cobalto sorgente", "cobalto sorgente")
	
	fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
	fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000)
	
	# Crea un istogramma con il numero di bin specificato
	hist_cobalto_sorgente=TH1F("cobalto sorgente", "cobalto sorgente", num_bin, 0, 8192)
	i=0
	# Riempie i bin con le altezze specificate
	for i in range(len(cobalto_sorgente)):
    		hist_cobalto_sorgente.Fill(cobalto_sorgente[i])


	# Adattamento della funzione ai dati
	hist_cobalto_sorgente.Fit("fit_function_Co", "ILR")


	hist_cobalto_sorgente.Draw()
	fit_function_Co.Draw("same")



	#parametri fit
	p0 = fit_function_Co.GetParameter(0)
	p0_err = fit_function_Co.GetParError(0)
	p1 = fit_function_Co.GetParameter(1)
	p1_err = fit_function_Co.GetParError(1)
	p2 = fit_function_Co.GetParameter(2)
	p2_err = fit_function_Co.GetParError(2)
	p3 = fit_function_Co.GetParameter(3)
	p3_err = fit_function_Co.GetParError(3)
	p4 = fit_function_Co.GetParameter(4)
	p4_err = fit_function_Co.GetParError(4)
	p5 = fit_function_Co.GetParameter(5)
	p5_err = fit_function_Co.GetParError(5)
	p6 = fit_function_Co.GetParameter(6)
	p6_err = fit_function_Co.GetParError(6)
	p7 = fit_function_Co.GetParameter(7)
	p7_err = fit_function_Co.GetParError(7)
	p8 = fit_function_Co.GetParameter(8)
	p8_err = fit_function_Co.GetParError(8)
	

	c_cobalto_sorgente.Draw()
	
	input()
	
	return p1, p1_err, p4, p4_err
