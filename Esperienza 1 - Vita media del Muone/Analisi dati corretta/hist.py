import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d

bin1=300
bin2=600



font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }

 


#VITA MEDIA
T= plb.loadtxt('vita_media.txt',unpack=True)
T_newsetup= plb.loadtxt('vita_media_newsetup.txt',unpack=True)


#*********************************************************************************************************************************************************************************************

#controllo eventi minori di 200ns


a=0
for i in range(len(T)):
	if (T[i]<=200e-9):
		a=a+1
		
		
print(a)

a=0
for i in range(len(T_newsetup)):
	if (T_newsetup[i]<=200e-9):
		a=a+1
		
		
print(a)


#***************************************************************************************************************************************************************************************************


f2=TF1("f2", "([0]*exp(-x/[1])) + [2]", 900e-9, 20e-6)
f2.SetParameters(821, 2.2e-6, 50)



c2=TCanvas('vita media new setup', 'vita media newsetup')

hist2=TH1F("vita media new setup", "Vita Media #mu", bin2, 0, 20e-6)
for i in range (len(T_newsetup)):
	hist2.Fill(T_newsetup[i])

hist2.Fit("f2", "ILR")

num_bin=bin2
entries = hist2.GetEntries()
ndof2 = f2.GetNDF()
chi_square2 = f2.GetChisquare()

p0_b = f2.GetParameter(0)
p0_err_b = f2.GetParError(0)
p1_b = f2.GetParameter(1)
p1_err_b = f2.GetParError(1)
p2_b = f2.GetParameter(2)
p2_err_b = f2.GetParError(2)



legend2 = TLegend(0.5, 0.5, 0.9, 0.8)
legend2.AddEntry(hist2, f"Histogram Entries: {entries:.0f}", "l")
legend2.AddEntry("", f"Number of Bins: {num_bin:.0f}", "")
legend2.AddEntry(f2, "Fit Function: A #times exp(-t/#tau)+C", "l")
legend2.AddEntry("", "Parameters:", "")
legend2.AddEntry("", f"A={p0_b:.0f}\pm{p0_err_b:.0f}", "")
legend2.AddEntry("", f"#tau = ({p1_b*(10**6):.2f} #pm {p1_err_b*(10**6):.2f}) #mu s", "")
legend2.AddEntry("", f"C = ({p2_b:.2f} #pm {p2_err_b:.2f}) ", "")
legend2.AddEntry("", f"#chi^{{2}} / ndof = {chi_square2:.0f} / {ndof2}", "")



hist2.GetXaxis().SetTitle("t [ns]")
hist2.GetYaxis().SetTitle("Eventi")
hist2.Draw()

f2.Draw("same")
legend2.Draw()
c2.Draw()

c2.SaveAs("hist2.root")



input()

