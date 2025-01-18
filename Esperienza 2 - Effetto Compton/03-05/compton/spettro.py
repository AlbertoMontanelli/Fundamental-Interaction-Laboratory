import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *


compton=plb.loadtxt('compton_15deg_35cm.dat', unpack=True)

#creo il canvas
c_compton = TCanvas("compton", "compton")

#decido i bin 
num_bin = len(compton)
# Crea un istogramma con il numero di bin specificato
hist_compton=TH1F("compton", "Spettro Compton - GATE:PMT01b&PMT02", num_bin, 0, num_bin)
i=0
# Riempie i bin con le altezze specificate
for i in range(num_bin):
	hist_compton.SetBinContent(i+1, compton[i])


hist_compton.Rebin(8)
hist_compton.GetXaxis().SetTitle("ch")
hist_compton.GetYaxis().SetTitle("Eventi")
hist_compton.Draw()

entries=0.
#entries
for i in range(len(compton)):
	entries=entries+compton[i]
	
num_bins = hist_compton.GetNbinsX()
	
# Aggiungi una legenda al plot
legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
legend_compton.AddEntry("","Data di acquisizione : 05/03/24, durata: 14h", "")
legend_compton.AddEntry(hist_compton, f"Histogram Entries: {entries:.0f}", "l")
legend_compton.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
legend_compton.AddEntry("", f"Angolo: 30 deg; Distanza:32.6cm", "")
legend_compton.AddEntry("", f"V_{{al}}=680V; V_{{thr}}=-20.7mV", "")
legend_compton.Draw()
	
c_compton.Draw()

input()
