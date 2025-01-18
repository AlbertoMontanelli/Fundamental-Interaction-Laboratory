import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from datetime import datetime
from scipy.interpolate import interp1d

dT_1e2=np.loadtxt("dT1e2_220cm.txt", unpack=True )
dT_2e3=np.loadtxt("dT2e3_220cm.txt", unpack=True)
dT_1e3=np.loadtxt("dT1e3_220cm.txt", unpack=True)


dT_1e2_mt=sorted(dT_1e2)
mediana_1e2=np.median(dT_1e2_mt)
tenperc_1e2=int(0.1*len(dT_1e2_mt))
dT_1e2_mt=dT_1e2_mt[tenperc_1e2:-tenperc_1e2]
mt_1e2=np.mean(dT_1e2_mt)

dT_2e3_mt=sorted(dT_2e3)
mediana_2e3=np.median(dT_2e3_mt)
tenperc_2e3=int(0.1*len(dT_2e3_mt))
dT_2e3_mt=dT_2e3_mt[tenperc_2e3:-tenperc_2e3]
mt_2e3=np.mean(dT_2e3_mt)

dT_1e3_mt=sorted(dT_1e3)
mediana_1e3=np.median(dT_1e3_mt)
tenperc_1e3=int(0.1*len(dT_1e3_mt))
dT_1e3_mt=dT_1e3_mt[tenperc_1e3:-tenperc_1e3]
mt_1e3=np.mean(dT_1e3_mt)

# Calcola la varianza della mediana utilizzando il bootstrap
def bootstrap_variance(data, num_bootstrap_samples=1000):
    medians = np.zeros(num_bootstrap_samples)
    for i in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        medians[i] = np.median(bootstrap_sample)
    return np.var(medians)

# weighted_means è l'array delle medie aritmetiche ponderate delle coppie di bin
err_mediana = np.sqrt(bootstrap_variance(dT_1e2))
err_mt=np.sqrt(bootstrap_variance(dT_1e2_mt))


num_bin = 200

#***************************************************************************************************************************************************************

# dT tra PMT01 e PMT02

#**************************************************************************************************************************************************************
c1=TCanvas("#Delta T tra PMT01 e PMT02", "#Delta T tra PMT01 e PMT02", 1600, 1200)

# Crea un istogramma con il numero di bin specificato
hist1=TH1F("#Delta T tra PMT01 e PMT02", "#Delta T tra PMT01 e PMT02", num_bin, -5, 20)
for i in range(len(dT_1e2)):
	hist1.Fill(dT_1e2[i])
	
fit_function_1 = TF1("fit_function_1", "gaus(0)+gaus(3)", 2, 15)
fit_function_1.SetParameters(157, 7, 0.44, 55, 7, 1.27)
hist1.Fit("fit_function_1", "ILR")

retta_mt_1 = TLine(mt_1e2, 0, mt_1e2, hist1.GetMaximum() )  
retta_mt_1.SetLineColor(kBlue)
retta_mt_1.SetLineWidth(2)

retta_mediana_1 = TLine(mediana_1e2, 0, mediana_1e2, hist1.GetMaximum() )  
retta_mediana_1.SetLineColor(kGreen)
retta_mediana_1.SetLineWidth(2)

hist1.GetXaxis().SetTitle("t [ns]")
hist1.GetYaxis().SetTitle("Conteggi")
hist1.Draw()
fit_function_1.Draw("same")
retta_mt_1.Draw("same")
retta_mediana_1.Draw("same")

entries = hist1.GetEntries()
ndof = fit_function_1.GetNDF()
chi_square = fit_function_1.GetChisquare()

p0_a = fit_function_1.GetParameter(0)
p0_err_a = fit_function_1.GetParError(0)
p1_a = fit_function_1.GetParameter(1)
p1_err_a = fit_function_1.GetParError(1)
p2_a = fit_function_1.GetParameter(2)
p2_err_a = fit_function_1.GetParError(2)
p3_a = fit_function_1.GetParameter(3)
p3_err_a = fit_function_1.GetParError(3)
p4_a = fit_function_1.GetParameter(4)
p4_err_a = fit_function_1.GetParError(4)
p5_a = fit_function_1.GetParameter(5)
p5_err_a = fit_function_1.GetParError(5)

#timestamp
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
timestamp_text = "Timestamp: " + current_time
latex = TLatex() # Creazione di un TLatex
latex.SetTextSize(0.03)
latex.SetNDC(True)  # Imposta le coordinate normalizzate del canvas
latex.DrawLatex(0.7, 0.05, timestamp_text) # Imposta la posizione del testo sul canvas (in questo caso nell'angolo in basso a destra)

legend1 = TLegend(0.5, 0.5, 0.9, 0.8)
legend1.AddEntry("", f" Data acquisizione: 17/04/2024", "")
legend1.AddEntry(hist1, f"Histogram Entries: {entries:.0f}", "l")
legend1.AddEntry(fit_function_1, "Fit Function: Gauss + Guass bkg", "l")
legend1.AddEntry(retta_mt_1, "Media troncata", "l")
legend1.AddEntry(retta_mediana_1, "Mediana", "l")
legend1.AddEntry("", f"Number of Bins: {num_bin:.0f}", "")
legend1.AddEntry("", "Parameters:", "")
legend1.AddEntry("", f"media troncata = ({mt_1e2:.2f} #pm{err_mt:.2f})ns", "")
legend1.AddEntry("", f"mediana = ({mediana_1e2:.2f} #pm {err_mediana:.2f}) ns", "")
legend1.AddEntry("", f"#mu = ({p1_a:.2f} #pm {p1_err_a:.2f}) ns, #sigma = ({abs(p2_a):.2f} #pm {p2_err_a:.2f}) ns ", "")
legend1.AddEntry("", f"#chi^{{2}} / ndof = {chi_square:.0f} / {ndof}", "")
legend1.AddEntry("", f"Posizione 220cm", "")

legend1.Draw()

c1.Draw()
c1.SaveAs("prova2.pdf")

a=np.array([p1_a, p1_err_a, abs(p2_a), p2_err_a, mediana_1e2, err_mediana, mt_1e2, err_mt])
#np.savetxt('medie_1e2_220cm.txt', a)



gApplication.Run()
