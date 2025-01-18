import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *

def compton_fit2(compton, fit_function_compton, rebinning):


 

	#creo il canvas
	c_compton = TCanvas("compton", "compton")

	#decido i bin 
	num_bin = len(compton)
	# Crea un istogramma con il numero di bin specificato
	hist_compton=TH1F("compton", "Spettro compton 27 marzo", num_bin, 0, num_bin)
	i=0
	# Riempie i bin con le altezze specificate
	for i in range(num_bin):
    		hist_compton.SetBinContent(i+1, compton[i])


	hist_compton.Rebin(rebinning)
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
	
	entries=0.
	#entries
	for i in range(len(compton)):
		entries=entries+compton[i]
	
	num_bins = hist_compton.GetNbinsX()	
	# Aggiungi una legenda al plot
	legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
	legend_compton.AddEntry("","Data di acquisizione : 28 Febbraio, #theta=20deg", "")
	legend_compton.AddEntry("","Durata acquisizione: 27 ore", "")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(hist_compton, f"Histogram Entries: {entries:.0f}", "l")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(fit_function_compton, f"Fit function: gauss_{{1}}+gauss_{{2}}+parabola", "l")
	# Aggiungi il numero di bin alla legenda
	legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
	# Aggiungi la riga "Parameters:" alla legenda
	legend_compton.AddEntry("", "Parameters:", "")
	# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
	legend_compton.AddEntry("", f"#mu_{{1}} = {p1_compton:.2f} #pm {p1_err_compton:.2f}, #sigma_{{1}} = {abs(p2_compton):.2f} #pm {p2_err_compton:.2f} ", "")
	legend_compton.AddEntry("", f"#mu_{{2}} = {p4_compton:.2f} #pm {p4_err_compton:.2f}, #sigma_{{2}} = {abs(p5_compton):.2f} #pm {p5_err_compton:.2f} ", "")
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

	input()

	return p1_compton, p1_err_compton, p4_compton, p4_err_compton
