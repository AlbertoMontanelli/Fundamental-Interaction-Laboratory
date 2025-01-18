import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *

def compton_fit(compton, fit_function_compton):


 

	#creo il canvas
	c_compton = TCanvas("compton", "compton")

	#decido i bin 
	num_bin = 1000
	# Crea un istogramma con il numero di bin specificato
	hist_compton=TH1F("compton", "Angolo 18deg 7 marzo", num_bin, 0, 8192)
	i=0
	# Riempie i bin con le altezze specificate
	for i in range(num_bin):
    		hist_compton.Fill(compton[i])


	# Adattamento della funzione ai dati
	hist_compton.Fit("fit_function_compton", "R")


	hist_compton.Draw()
	fit_function_compton.Draw("same")



	#chi quadro/gradi di libertà
	ndof = fit_function_compton.GetNDF()
	chi_square = fit_function_compton.GetChisquare()
	#numero bin
	num_bins = hist_compton.GetNbinsX()
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
	

	c_compton.Draw()

	
	return p1_compton, p1_err_compton, p4_compton, p4_err_compton
