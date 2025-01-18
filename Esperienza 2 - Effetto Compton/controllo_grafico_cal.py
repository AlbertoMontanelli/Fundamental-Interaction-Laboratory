import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *



#*********************************************************

# CONVERSIONE CANALI -> ENERGIA

#********************************************************


ch=np.array([5683.78048931, 6499.63057009, 2562.42702644, 6189.20934599, 3243.17411403])
dch=np.array([0.83194746, 0.61214053, 0.50277544, 2.80615205, 0.10788366])


E=np.array([1.173, 1.333, 0.511, 1.275, 0.662])
linear = TF1("linear", "pol1(0)", 0, 2)
linear.SetParameters(216, 4586) 
parabola = TF1("parabola", "pol2(0)", 0, 2)
parabola.SetParameters(-256,5695, -588) 
c_cal = TCanvas("calibrazione", "calibrazione")

graph =TGraphErrors(5, E, ch, nullptr, dch)


# Imposta lo stile del grafico
graph.SetLineColor(kBlack)
graph.SetMarkerStyle(20)
graph.SetMarkerSize(1)
graph.SetFillStyle(0)
graph.SetFillColor(0)

# Aggiungi le label agli assi
graph.GetXaxis().SetTitle("E [MeV]")
graph.GetYaxis().SetTitle("ch")
	
graph.Fit(linear, "S")
graph.Fit(parabola, "S")
	
# Disegna il grafico con i punti e le linee dei fit
graph.Draw("AP")

# Disegna la funzione del primo fit
linear.SetLineColor(kBlue)
linear.Draw("same")
	
# Disegna la funzione del secondo fit
parabola.SetLineColor(kRed)
parabola.Draw("same")
	
	
	
# Estrai i parametri del fit
m = linear.GetParameter(0)
m_err = linear.GetParError(0)
q = linear.GetParameter(1)
q_err = linear.GetParError(1)
	
c = parabola.GetParameter(0)
c_err = parabola.GetParError(0)
b = parabola.GetParameter(1)
b_err = parabola.GetParError(1)
a = parabola.GetParameter(2)
a_err = parabola.GetParError(2)
	
	
	
# Visualizza il canvas
c_cal.Draw()


input()	
