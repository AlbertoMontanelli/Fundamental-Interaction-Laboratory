import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
from ROOT import *


 


fit_function_Am = TF1("fit_function_Am", "gaus(0)+pol2(3)", 152, 464)
fit_function_Am.SetParameters(379824, 311, 21, -10197, 123, -0.21)


def fit_cobalto(cobalto, num_bins):

	c_Co=TCanvas("cobalto", "Cobalto")
	fit_function_Co = TF1("fit_function_Co", "gaus(0)+gaus(3)+gaus(6)", 5000, 7500)
	fit_function_Co.SetParameters(3500, 5585, 121, 3009, 6313, 137, 1303, 4499, 1000) 
	fit_function_Co.SetLineWidth(1)
	#decido i bin 
	num_bin = num_bins
	# Crea un istogramma con il numero di bin specificato
	hist_cobalto=TH1F("cobalto", "Sorgente di Cobalto", num_bin, 0, 8192)
	for i in range(len(cobalto)):
    		hist_cobalto.Fill(cobalto[i])
	
	# Adattamento della funzione ai dati
	hist_cobalto.Fit("fit_function_Co", "ILR")
	hist_cobalto.GetXaxis().SetTitle("ch")
	hist_cobalto.GetYaxis().SetTitle("Eventi")
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
	
	num_bins = hist_cobalto.GetNbinsX()
	num_entries = hist_cobalto.GetEntries()
	ndof = fit_function_Co.GetNDF()
	chi_square = fit_function_Co.GetChisquare()
	# Aggiungi una legenda al plot
	legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
	legend_compton.AddEntry("","Data di acquisizione : 19 Marzo", "")
	legend_compton.AddEntry("","Durata acquisizione: 5min", "")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(hist_cobalto, f"Histogram Entries: {num_entries:.0f}", "l")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(fit_function_Co, f"Fit function: gauss_{{1}}+gauss_{{2}}+parabola", "l")
	# Aggiungi il numero di bin alla legenda
	legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
	# Aggiungi la riga "Parameters:" alla legenda
	legend_compton.AddEntry("", "Parameters:", "")
	# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
	legend_compton.AddEntry("", f"#mu_{{1}} = {p1_Co:.2f} #pm {p1_err_Co:.2f}, #sigma_{{1}} = {abs(p2_Co):.2f} #pm {p2_err_Co:.2f} ", "")
	legend_compton.AddEntry("", f"#mu_{{2}} = {p4_Co:.2f} #pm {p4_err_Co:.2f}, #sigma_{{2}} = {abs(p5_Co):.2f} #pm {p5_err_Co:.2f} ", "")
	legend_compton.AddEntry("", f"#chi^{{2}}/ndof = {chi_square:.0f}/{ndof:.0f} ", "")
	# Disegna la legenda
	legend_compton.Draw()
	
	c_Co.Draw()
	input()
	return p1_Co, p1_err_Co, p4_Co, p4_err_Co
	

def fit_sodio(sodio, num_bins):
	c_Na=TCanvas("sodio", "Sorgente di Sodio")
	fit_function_Na = TF1("fit_function_Na", "gaus(0)+pol2(3)", 2000, 3000)
	fit_function_Na.SetParameters(1600, 2500, 89,  -400, 0.620508, -0.000104099)
	fit_function_Na.SetLineWidth(1)
	fit_function_Na2 = TF1("fit_function_Na2", "gaus(0)+pol1(3)", 5600, 6800)
	fit_function_Na2.SetParameters(126, 6362, 134,  83.1, -0.007401)
	fit_function_Na2.SetLineWidth(1)
	#decido i bin 
	num_bin = num_bins
	# Crea un istogramma con il numero di bin specificato
	hist_sodio=TH1F("sodio", "Sorgente di Sodio", num_bin, 0, 8192)
	for i in range(len(sodio)):
    		hist_sodio.Fill(sodio[i])

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

	hist_sodio.GetXaxis().SetTitle("ch")
	hist_sodio.GetYaxis().SetTitle("Eventi")
	
	hist_sodio.Draw()
	fit_function_Na.Draw("same")
	fit_function_Na2.Draw("same")
	
	
	num_bins = hist_sodio.GetNbinsX()
	num_entries = hist_sodio.GetEntries()
	ndof = fit_function_Na.GetNDF()
	chi_square = fit_function_Na.GetChisquare()
	ndof2 = fit_function_Na2.GetNDF()
	chi_square2 = fit_function_Na2.GetChisquare()
	# Aggiungi una legenda al plot
	legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
	legend_compton.AddEntry("","Data di acquisizione : 19 Marzo", "")
	legend_compton.AddEntry("","Durata acquisizione: 5min", "")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(hist_sodio, f"Histogram Entries: {num_entries:.0f}", "l")
	legend_compton.AddEntry(fit_function_Na, f"Fit function: gauss_{{1}}+parabola", "l")
	legend_compton.AddEntry(fit_function_Na2, f"Fit function: gauss_{{2}}+retta", "l")
	# Aggiungi il numero di bin alla legenda
	legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
	# Aggiungi la riga "Parameters:" alla legenda
	legend_compton.AddEntry("", "Parameters:", "")
	# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
	legend_compton.AddEntry("", f"#mu_{{1}} = {p1_Na:.2f} #pm {p1_err_Na:.2f}, #sigma_{{1}} = {abs(p2_Na):.2f} #pm {p2_err_Na:.2f} ", "")
	legend_compton.AddEntry("", f"#mu_{{2}} = {p1_Na_2:.2f} #pm {p1_err_Na_2:.2f}, #sigma_{{2}} = {abs(p2_Na_2):.2f} #pm {p2_err_Na_2:.2f} ", "")
	legend_compton.AddEntry("", f"#chi_{{1}}^{{2}}/ndof_{{1}} = {chi_square:.0f}/{ndof:.0f} ", "")
	legend_compton.AddEntry("", f"#chi_{{2}}^{{2}}/ndof_{{2}} = {chi_square2:.0f}/{ndof2:.0f} ", "")
	# Disegna la legenda
	legend_compton.Draw()
	
	
	c_Na.Draw()
	input()
	return p1_Na, p1_err_Na, p1_Na_2, p1_err_Na_2
	
	
def fit_cesio(cesio, num_bins):
	c_Cs=TCanvas("cesio", "Sorgenti di Cesio")
	fit_function_Cs = TF1("fit_function_Cs", "gaus(0)+pol2(3)", 2600, 4016)
	fit_function_Cs.SetParameters(10373, 3450, -104, 5100, -2.55, 0.0003086)
	fit_function_Cs.SetLineWidth(1)
	#decido i bin 
	num_bin = num_bins
	# Crea un istogramma con il numero di bin specificato
	hist_cesio=TH1F("cesio", "Sorgente di Cesio", num_bin, 0, 8192)
	for i in range(len(cesio)):
    		hist_cesio.Fill(cesio[i])
    	
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

	hist_cesio.GetXaxis().SetTitle("ch")
	hist_cesio.GetYaxis().SetTitle("Eventi")
	
	hist_cesio.Draw()
	fit_function_Cs.Draw("same")
	
	num_bins = hist_cesio.GetNbinsX()
	num_entries = hist_cesio.GetEntries()
	ndof = fit_function_Cs.GetNDF()
	chi_square = fit_function_Cs.GetChisquare()
	# Aggiungi una legenda al plot
	legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
	legend_compton.AddEntry("","Data di acquisizione : 19 Marzo", "")
	legend_compton.AddEntry("","Durata acquisizione: 5min", "")
	# Aggiungi il numero di entries dell'istogramma alla legenda
	legend_compton.AddEntry(hist_cesio, f"Histogram Entries: {num_entries:.0f}", "l")
	legend_compton.AddEntry(fit_function_Cs, f"Fit function: gauss_{{1}}+parabola", "l")
	# Aggiungi il numero di bin alla legenda
	legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
	# Aggiungi la riga "Parameters:" alla legenda
	legend_compton.AddEntry("", "Parameters:", "")
	# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
	legend_compton.AddEntry("", f"#mu_{{1}} = {p1_Cs:.2f} #pm {p1_err_Cs:.2f}, #sigma_{{1}} = {abs(p2_Cs):.2f} #pm {p2_err_Cs:.2f} ", "")
	legend_compton.AddEntry("", f"#chi^{{2}}/ndof = {chi_square:.0f}/{ndof:.0f} ", "")
	# Disegna la legenda
	legend_compton.Draw()	
	
	
	c_Cs.Draw()
	input()
	return p1_Cs, p1_err_Cs
	
	

	
	
	


