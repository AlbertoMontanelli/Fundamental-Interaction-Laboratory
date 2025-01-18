import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *


"""
PYTHON 


# Definizione della funzione della parabola
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

# Definizione della funzione della retta
def retta(x, m, q):
    return m * x + q

# Fit dei dati con la parabola e la retta
popt_parabola, pcov_parabola = curve_fit(parabola, x, y, sigma=dy)
popt_retta, pcov_retta = curve_fit(retta, x, y, sigma=dy)

# Estraiamo gli errori dei parametri
perr_parabola = np.sqrt(np.diag(pcov_parabola))
perr_retta = np.sqrt(np.diag(pcov_retta))

# Plot dei dati e dei fit
plt.errorbar(x, y, yerr=dy, fmt='o', label='Dati')

x_range = np.linspace(min(x), max(x), 100)

plt.plot(x_range, parabola(x_range, *popt_parabola), 'r-', label='Parabola fit: a=%5.3f ± %5.3f, b=%5.3f ± %5.3f, c=%5.3f ± %5.3f' % (*popt_parabola, *perr_parabola))
plt.plot(x_range, retta(x_range, *popt_retta), 'g--', label='Retta fit: m=%5.3f ± %5.3f, q=%5.3f ± %5.3f' % (*popt_retta, *perr_retta))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
"""

# da fit_sorgenti_controllo.py
ch=np.loadtxt('dati/ch_cal_19marzo_iniziofinemedia.txt', unpack=True)
dch=np.loadtxt('dati/dch_cal_19marzo_iniziofinemedia.txt', unpack=True)

"""
ch1=ch[:,0]
ch2=ch[:,1]
dch1=dch[:,0]
dch2=dch[:,1]
"""
 
E=np.array([1.173, 1.333, 0.511, 1.275, 0.662])


"""
c_cal = TCanvas("calibrazione", "calibrazione")
c_cal.Divide(2,1)

linear1 = TF1("linear1", "pol1", 0, 2)
linear1.SetParameters(216, 4586) 
parabola1 = TF1("parabola1", "pol2", 0, 2)
parabola1.SetParameters(-256, 5695, -588) 

linear2 = TF1("linear2", "pol1", 0, 2)
linear2.SetParameters(216, 4586) 
parabola2 = TF1("parabola2", "pol2", 0, 2)
parabola2.SetParameters(-256, 5695, -588)

c_cal.cd(1)
graph1 = TGraphErrors(5, E, ch1, nullptr, dch1)
graph1.SetTitle("Calibrazione inizio 13 marzo")
graph1.SetLineColor(kBlack)
graph1.SetMarkerStyle(20)
graph1.SetMarkerSize(1)
graph1.GetXaxis().SetTitle("E [MeV]")
graph1.GetYaxis().SetTitle("ch")    
graph1.Fit(linear1, "S")
graph1.Fit(parabola1, "S") 
graph1.Draw("AP")
linear1.SetLineColor(kBlue)
linear1.Draw("same")    
parabola1.SetLineColor(kRed)
parabola1.Draw("same")


c_cal.cd(2)
graph2 = TGraphErrors(5, E, ch2, nullptr, dch2)
graph2.SetTitle("Calibrazione fine 13 marzo")
graph2.SetLineColor(kBlack)
graph2.SetMarkerStyle(20)
graph2.SetMarkerSize(1)
graph2.GetXaxis().SetTitle("E [MeV]")
graph2.GetYaxis().SetTitle("ch")    
graph2.Fit(linear2, "S")
graph2.Fit(parabola2, "S") 
graph2.Draw("AP")
linear2.SetLineColor(kGreen)
linear2.Draw("same")    
parabola2.SetLineColor(kMagenta)
parabola2.Draw("same")

c_cal.Draw()


"""
c_cal = TCanvas("calibrazione", "calibrazione")
c_cal.SetGrid()

linear1 = TF1("linear", "pol1(0)", 0, 2)
linear1.SetParameters(216, 4586) 
parabola1 = TF1("parabola", "pol2(0)", 0, 2)
parabola1.SetParameters(-256,5695, -588) 



graph1 =TGraphErrors(5, E, ch, nullptr, dch)



# Imposta lo stile del grafico
graph1.SetLineColor(kBlack)
graph1.SetTitle("Calibrazione mediata 19 Marzo")
graph1.SetMarkerStyle(20)
graph1.SetMarkerSize(1)
# Aggiungi le label agli assi
graph1.GetXaxis().SetTitle("E [MeV]")
graph1.GetYaxis().SetTitle("ch")	
# Effettua i fit e ottieni l'oggetto TFitResult
result_linear1 = graph1.Fit(linear1, "S")
result_parabola1 = graph1.Fit(parabola1, "S")	
# Disegna il grafico con i punti e le linee dei fit
graph1.Draw("AP")
# Disegna la funzione del primo fit
linear1.SetLineColor(kBlue)
linear1.SetLineWidth(1)
linear1.Draw("same")
	
# Disegna la funzione del secondo fit
parabola1.SetLineColor(kGreen)
parabola1.SetLineWidth(1)
parabola1.Draw("same")

graph1.Draw("P same");
# Estrai i parametri del fit
m1 = linear1.GetParameter(1)
m_err1 = linear1.GetParError(1)
q1 = linear1.GetParameter(0)
q_err1 = linear1.GetParError(0)
c1 = parabola1.GetParameter(0)
c_err1 = parabola1.GetParError(0)
b1 = parabola1.GetParameter(1)
b_err1 = parabola1.GetParError(1)
a1 = parabola1.GetParameter(2)
a_err1 = parabola1.GetParError(2)

chi_square_value1 = linear1.GetChisquare()
chi_square_value2 = parabola1.GetChisquare()

# Aggiungi una legenda al plot
legend_compton = TLegend(0.7, 0.6, 0.98, 0.88)
legend_compton.SetTextSize(0.025)  # Imposta la dimensione del font desiderata (cambia il valore a tuo piacimento)
# Aggiungi il numero di entries dell'istogramma alla legenda
legend_compton.AddEntry(linear1, f"Fit function: mx+q", "l")
legend_compton.AddEntry(parabola1, f"Fit function: ax^{{2}}+bx+c", "l")
# Aggiungi la riga "Parameters:" alla legenda
legend_compton.AddEntry("", "Parameters:", "")
# Aggiungi i parametri della funzione alla legenda con i loro valori e errori
legend_compton.AddEntry("", f"m = ({m1:.0f} #pm {m_err1:.0f}) MeV^{{-1}}, q = {q1:.0f} #pm {q_err1:.0f} ", "")
legend_compton.AddEntry("", f"a = ({a1:.0f} #pm {a_err1:.0f}) MeV^{{-2}}, b = ({b1:.0f} #pm {b_err1:.0f}) MeV^{{-1}}, c = {c1:.0f} #pm {c_err1:.0f} ", "")
legend_compton.AddEntry("", f"#chi^{{2}} retta= {chi_square_value1:.0f}, #chi^{{2}} parabola = {chi_square_value2:.0f}", "")
# Disegna la legenda
legend_compton.Draw()	



		
	

# Visualizza il canvas
c_cal.Draw()

input()
#************************************************

# PARAMETRI FIT

#*************************************************



"""
# Estrai i parametri del fit
m2 = linear2.GetParameter(1)
m_err2 = linear2.GetParError(1)
q2 = linear2.GetParameter(0)
q_err2 = linear2.GetParError(0)
c2 = parabola2.GetParameter(0)
c_err2 = parabola2.GetParError(0)
b2 = parabola2.GetParameter(1)
b_err2 = parabola2.GetParError(1)
a2 = parabola2.GetParameter(2)
a_err2 = parabola2.GetParError(2)

"""
# Ottieni le matrici di covarianza dai risultati del fit
cov_matrix_linear1 = result_linear1.GetCovarianceMatrix()
cov_matrix_parabola1 = result_parabola1.GetCovarianceMatrix()

# Stampa le matrici di covarianza
print("Matrice di covarianza per il fit lineare:")
cov_matrix_linear1.Print()
print("Matrice di covarianza per il fit parabolico:")
cov_matrix_parabola1.Print()

"""
pars1=np.array([m1, q1, c1, b1, a1])
np.savetxt('dati/pars_cal_12e13marzo_iniziofinemedia.txt', pars1)
dpars1=np.array([m_err1, q_err1, c_err1, b_err1, a_err1])
np.savetxt('dati/err_pars_cal_12e13marzo_iniziofinemedia.txt', dpars1) 

print(pars1)
print(dpars1)




# Converti la matrice di covarianza in un array numpy
cov_matrix_linear1_array = np.array([[cov_matrix_linear1(i, j) for j in range(cov_matrix_linear1.GetNcols())] for i in range(cov_matrix_linear1.GetNrows())])
cov_matrix_parabola1_array = np.array([[cov_matrix_parabola1(i, j) for j in range(cov_matrix_parabola1.GetNcols())] for i in range(cov_matrix_parabola1.GetNrows())])

# Salva le matrici di covarianza in file di testo
np.savetxt('dati/cov_lineare_12e13marzo_iniziofinemedia.txt', cov_matrix_linear1_array)
np.savetxt('dati/cov_parabola_12e13marzo_iniziofinemedia.txt', cov_matrix_parabola1_array)

pars2=np.array([m2, q2, c2, b2, a2])
dpars2=np.array([m_err2, q_err2, c_err2, b_err2, a_err2])



np.savetxt('dati/err_pars_cal_19marzo_inizio.txt', dpars1) 
np.savetxt('dati/pars_cal_19marzo_inizio.txt', pars1)
np.savetxt('dati/err_pars_cal_19marzo_fine.txt', dpars2) 
np.savetxt('dati/pars_cal_19marzo_fine.txt', pars2)

"""
input()


