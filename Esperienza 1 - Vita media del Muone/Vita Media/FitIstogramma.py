import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
from ROOT import *


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }


def dEff(n,N):
	deff = np.sqrt(n*(1-(n/N)))/N
	return deff
	
def dV(V):
	dV=V*(0.05/100)+1
	return dV
	
#def exp(t, tau, A, offset):
#	return A*plb.exp(-t/tau)+offset
    


#VITA MEDIA
T= plb.loadtxt('MV_acquisizione12-12-20us-clean.txt',unpack=True)
Tmin=min(T)
Tmax=max(T)




c=TCanvas('boh', 'boh', 1200, 1000)


c.Divide(2)
c.cd(1)


i=0			
hist=TH1F("Parametri", "Vita Media #mu", 200, 0, 20e-6)
for i in range (len(T)):
	hist.Fill(T[i])


hist1=hist.Clone()
hist1.GetXaxis().SetTitle("Tempo [s]")


f1=TF1("f1", "([0]*exp(-x/[1])) + [2]")
f1.SetParameters(200, 2.2e-6, 1)	
hist.Fit("f1", "", "", 1.2e-6, 20e-6)
p0 = f1.GetParameter(0)
p0_err = f1.GetParError(0)
p1 = f1.GetParameter(1)
p1_err = f1.GetParError(1)
p2 = f1.GetParameter(2)
p2_err = f1.GetParError(2)

hist.GetXaxis().SetTitle("Tempo [s]")
hist.GetYaxis().SetTitle("Eventi")

hist1.Fit("f1", "", "", 1.2e-6, 20e-6)


ndof = f1.GetNDF()
chi_square = f1.GetChisquare()
entries = hist.GetEntries()
num_bins = hist.GetNbinsX()



hist.Draw()
f1.Draw("same")

# Aggiungi una legenda al plot
legend = TLegend(0.6, 0.6, 0.88, 0.88)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend.AddEntry(hist, f"Histogram Entries: {entries:.0f}", "l")
# Aggiungi il numero di bin alla legenda
legend.AddEntry("", f"Number of Bins: {num_bins:.0f}", "")
# Aggiungi la funzione di fit alla legenda con la sua forma matematica
legend.AddEntry(f1, "Fit Function: A*e^{-x / #tau} + C", "l")
# Aggiungi la riga "Parameters:" alla legenda
legend.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend.AddEntry("", f"A = {p0:.0f} #pm {p0_err:.0f}", "")
legend.AddEntry("", f"#tau = (2.22 #pm 0.04) #mus", "")
legend.AddEntry("", f"C = {p2:.0f} #pm {p2_err:.0f}", "")
# Aggiungi il valore di chi quadro su gradi di libertà alla legenda
legend.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")


# Disegna la legenda
legend.Draw()




# Aggiungi la legenda al canvas
gPad.Update()
gPad.Modified()


c.cd(2)
hist1.Draw("E")
f1.Draw("same")


# Mostra il canvas
c.Draw()


"""
outfile=TFile('F1.ROOT', 'RECREATE')
hist.Write('hist_25us_unclean')
outfile.Close()
"""


input()

