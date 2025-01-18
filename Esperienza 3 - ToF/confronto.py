import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d
import simpy


"""
#v_rel=np.loadtxt('v_rel_alberto.txt', unpack=True)
v_nonrel=np.loadtxt('v_nonrel_alberto.txt', unpack=True)
v_rel=np.loadtxt('v_decay_alberto.txt', unpack=True)

c = TCanvas("Confronto Istogrammi", "Confronto Istogrammi")

h2=TH1F("h2 ", " Confronto 1/#beta delle particelle fermate/non fermate nel Pb; 1/#beta; eventi ", 250, -1.5, 3)
for i in range(len(v_rel)):
	h2.Fill(v_rel[i])

max_bin_content2 = h2.GetBinContent(h2.GetMaximumBin())

if max_bin_content2 != 0:
    h2.Scale(1.0 / h2.GetEntries())

h1=TH1F("h1 ", "Confronto 1/#beta delle particelle delle particelle fermate/non fermate nel Pb; 1/#beta; eventi   ", 250, -1.5, 3)
for i in range(len(v_nonrel)):
	h1.Fill(v_nonrel[i])

max_bin_content1 = h1.GetBinContent(h1.GetMaximumBin())

if max_bin_content1 != 0:
    h1.Scale(1.0 / h1.GetEntries())




h1.SetLineColor(kRed)
h1.Draw('HIST')
#h1.SetLineWidth(2)    
h2.SetLineColor(kBlue)
#h2.SetLineWidth(2)
h2.Draw('HIST SAME')


#timestamp
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
timestamp_text = "Timestamp: " + current_time
latex = TLatex() # Creazione di un TLatex
latex.SetTextSize(0.03)
latex.SetNDC(True)  # Imposta le coordinate normalizzate del canvas
latex.DrawLatex(0.7, 0.05, timestamp_text) # Imposta la posizione del testo sul canvas (in questo caso nell'angolo in basso a destra)

# Aggiungere una legenda
legend = TLegend(0.7, 0.8, 0.9, 0.9)
legend.AddEntry(h1, "Trigger DRS: 1&2&3&#bar{4}", "l")   
legend.AddEntry(h2, "Trigger DRS: 1&2&3", "l")
legend.AddEntry("", f"bin: 250", "")
legend.Draw()

chi2 = h1.Chi2Test(h2, "WW CHI2")
print("Chi-Quadro: ", chi2)
p_value = h1.Chi2Test(h2, "WW CHI2 P")
print("p-value: ", p_value)

c.Draw()

input()
"""
