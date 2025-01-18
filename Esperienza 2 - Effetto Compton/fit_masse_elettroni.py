import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
import math
#from ROOT import *


dtheta=np.array([np.pi/180, np.pi/180, np.pi/180, np.pi/180, np.pi/180])
dtheta = np.sqrt((dtheta*np.sqrt(1/12))**2 +  (0.006590032400667419)**2)

print(dtheta)

Ea_lin1, Eb_lin1=np.loadtxt('dati/E_finali_lin_28e29febbraio_iniziofinemedia.txt', unpack=True)
dEa_lin1, dEb_lin1=np.loadtxt('dati/E_finali_lin_err_28e29febbraio_iniziofinemedia.txt', unpack=True)
Ea_pol1, Eb_pol1=np.loadtxt('dati/E_finali_pol_28e29febbraio_iniziofinemedia.txt', unpack=True)
dEa_pol1, dEb_pol1=np.loadtxt('dati/E_finali_pol_err_28e29febbraio_iniziofinemedia.txt', unpack=True)
theta1=math.radians(20)
#theta1=20

Ea_lin2, Eb_lin2=np.loadtxt('dati/E_finali_lin_5e6marzo_iniziofinemedia.txt', unpack=True)
dEa_lin2, dEb_lin2=np.loadtxt('dati/E_finali_lin_err_5e6marzo_iniziofinemedia.txt', unpack=True)
Ea_pol2, Eb_pol2=np.loadtxt('dati/E_finali_pol_5e6marzo_iniziofinemedia.txt', unpack=True)
dEa_pol2, dEb_pol2=np.loadtxt('dati/E_finali_pol_err_5e6marzo_iniziofinemedia.txt', unpack=True)
theta2=math.radians(15)
#theta2=15

Ea_lin3, Eb_lin3=np.loadtxt('dati/E_finali_lin_7marzo_iniziofinemedia.txt', unpack=True)
dEa_lin3, dEb_lin3=np.loadtxt('dati/E_finali_lin_err_7marzo_iniziofinemedia.txt', unpack=True)
Ea_pol3, Eb_pol3=np.loadtxt('dati/E_finali_pol_7marzo_iniziofinemedia.txt', unpack=True)
dEa_pol3, dEb_pol3=np.loadtxt('dati/E_finali_pol_err_7marzo_iniziofinemedia.txt', unpack=True)
theta3=math.radians(18)
#theta3=18

Ea_lin4, Eb_lin4=np.loadtxt('dati/E_finali_lin_12e13marzo_iniziofinemedia.txt', unpack=True)
dEa_lin4, dEb_lin4=np.loadtxt('dati/E_finali_lin_err_12e13marzo_iniziofinemedia.txt', unpack=True)
Ea_pol4, Eb_pol4=np.loadtxt('dati/E_finali_pol_12e13marzo_iniziofinemedia.txt', unpack=True)
dEa_pol4, dEb_pol4=np.loadtxt('dati/E_finali_pol_err_12e13marzo_iniziofinemedia.txt', unpack=True)
theta4=math.radians(15)
#theta4=15

Ea_lin5, Eb_lin5=np.loadtxt('dati/E_finali_lin_19marzo_iniziofinemedia.txt', unpack=True)
dEa_lin5, dEb_lin5=np.loadtxt('dati/E_finali_lin_err_19marzo_iniziofinemedia.txt', unpack=True)
Ea_pol5, Eb_pol5=np.loadtxt('dati/E_finali_pol_19marzo_iniziofinemedia.txt', unpack=True)
dEa_pol5, dEb_pol5=np.loadtxt('dati/E_finali_pol_err_19marzo_iniziofinemedia.txt', unpack=True)
theta5=math.radians(19)
#theta5=15

Ea_lin6, Eb_lin6=np.loadtxt('dati/E_finali_lin_19e20marzo_iniziofinemedia.txt', unpack=True)
dEa_lin6, dEb_lin6=np.loadtxt('dati/E_finali_lin_err_19e20marzo_iniziofinemedia.txt', unpack=True)
Ea_pol6, Eb_pol6=np.loadtxt('dati/E_finali_pol_19e20marzo_iniziofinemedia.txt', unpack=True)
dEa_pol6, dEb_pol6=np.loadtxt('dati/E_finali_pol_err_19e20marzo_iniziofinemedia.txt', unpack=True)
theta6=math.radians(19)
#theta6=19

Ea_lin7, Eb_lin7=np.loadtxt('dati/E_finali_lin_20e21marzo_iniziofinemedia.txt', unpack=True)
dEa_lin7, dEb_lin7=np.loadtxt('dati/E_finali_lin_err_20e21marzo_iniziofinemedia.txt', unpack=True)
Ea_pol7, Eb_pol7=np.loadtxt('dati/E_finali_pol_20e21marzo_iniziofinemedia.txt', unpack=True)
dEa_pol7, dEb_pol7=np.loadtxt('dati/E_finali_pol_err_20e21marzo_iniziofinemedia.txt', unpack=True)
theta7=math.radians(22)
#theta7=22

Ea_lin8, Eb_lin8=np.loadtxt('dati/E_finali_lin_21marzo_iniziofinemedia.txt', unpack=True)
dEa_lin8, dEb_lin8=np.loadtxt('dati/E_finali_lin_err_21marzo_iniziofinemedia.txt', unpack=True)
Ea_pol8, Eb_pol8=np.loadtxt('dati/E_finali_pol_21marzo_iniziofinemedia.txt', unpack=True)
dEa_pol8, dEb_pol8=np.loadtxt('dati/E_finali_pol_err_21marzo_iniziofinemedia.txt', unpack=True)
theta8=math.radians(15)
#theta8=15


theta=np.array([theta4, theta3, theta6, theta1, theta7])

Ea_lin=np.array([Ea_lin2, Ea_lin3, Ea_lin6, Ea_lin1, Ea_lin7])
dEa_lin=np.array([dEa_lin2, dEa_lin3, dEa_lin6, dEa_lin1, dEa_lin7])
Eb_lin=np.array([Eb_lin2, Eb_lin3, Eb_lin6, Eb_lin1, Eb_lin7])
dEb_lin=np.array([dEb_lin2, dEb_lin3, dEb_lin6, dEb_lin1, dEb_lin7])

Ea_pol=np.array([Ea_pol2, Ea_pol3, Ea_pol6, Ea_pol1, Ea_pol7])
dEa_pol=np.array([dEa_pol2, dEa_pol3, dEa_pol6, dEa_pol1, dEa_pol7])
Eb_pol=np.array([Eb_pol2, Eb_pol3, Eb_pol6, Eb_pol1, Eb_pol7])
dEb_pol=np.array([dEb_pol2, dEb_pol3, dEb_pol6, dEb_pol1, dEb_pol7])

#****************************************************************************************************************************************

# FIT

#*******************************************************************************************************************************************


def E_fin_a(theta, m):
    return 1.172/(1+(1.172/m)*(1-np.cos(theta-math.radians(93-92.9536862495961))))

def E_fin_b(theta, m):
    return 1.333/(1+(1.333/m)*(1-np.cos(theta-math.radians(93-92.9536862495961))))



#******************************************************************************************************************

font = {'family': 'serif', 'weight': 'normal', 'size': 45, 'color': 'red'}

#dEa_pol = 1.21*dEa_pol
# Fai il fit dei dati sperimentali
popt, pcov = curve_fit(E_fin_a, theta, Ea_pol, sigma=dEa_pol, absolute_sigma=False)

# Parametri ottimizzati
m_opt = popt
error_m_opt = np.sqrt(np.diag(pcov))

# Genera dati per il plot
theta_fit = np.linspace(min(theta), max(theta), 1000)
E_fit = E_fin_a(theta_fit, m_opt)


# Calcola i valori predetti dalla funzione di fit
y_pred = E_fin_a(theta, *popt)

# Calcola il residuo
residuals = Ea_pol - y_pred


# Calcola il chi-quadro
chi_square3 = np.sum((residuals / dEa_pol)**2)

print("Chi-quadro picco 1 pol", chi_square3)

# Disegna il plot
plt.figure('Picco 1 POL')
plt.errorbar(theta, Ea_pol, xerr=dtheta, yerr=dEa_pol, fmt='o')
plt.plot(theta_fit, E_fit, 'r-')
plt.xlabel(r'$\theta$ [rad]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.ylabel(r'$E_{fin}$ [MeV]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})

legenda = r"$m_e =({:.0f}\pm{:.0f})$keV".format(m_opt[0] * 1000, error_m_opt[0] * 1000)
plt.text(0.5, 0.9, legenda, transform=plt.gca().transAxes, fontdict=font, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.title(r'Energia in funzione di $\theta$, $E_{in}$ = 1.172 MeV',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.grid(True)

# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/Fit1.pdf', dpi=1200, bbox_inches='tight')


#**********************************************************************************************************************************************

#dEb_pol = 1.53*dEb_pol
# Fai il fit dei dati sperimentali
popt, pcov = curve_fit(E_fin_b, theta, Eb_pol, sigma=dEb_pol, absolute_sigma=False)

# Parametri ottimizzati
m_opt = popt
error_m_opt = np.sqrt(np.diag(pcov))

# Genera dati per il plot
theta_fit = np.linspace(min(theta), max(theta), 1000)
E_fit = E_fin_b(theta_fit,m_opt)

# Calcola i valori predetti dalla funzione di fit
y_pred = E_fin_b(theta, *popt)

# Calcola il residuo
residuals = Eb_pol - y_pred


# Calcola il chi-quadro
chi_square4 = np.sum((residuals / dEb_pol)**2)

print("Chi-quadro picco 2 pol:", chi_square4)



# Disegna il plot
plt.figure("picco 2 POL")
plt.errorbar(theta, Eb_pol, xerr=dtheta, yerr=dEb_pol, fmt='o')
plt.plot(theta_fit, E_fit, 'r-')
plt.xlabel(r'$\theta$ [rad]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.ylabel(r'$E_{fin}$ [MeV]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})

legenda = r"$m_e =({:.0f}\pm{:.0f})$keV".format(m_opt[0] * 1000, error_m_opt[0] * 1000)
plt.text(0.5, 0.9, legenda, transform=plt.gca().transAxes, fontdict=font, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.title(r'Energia in funzione di $\theta$, $E_{in}$ = 1.333 MeV',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.grid(True)

# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/Fit2.pdf', dpi=1200, bbox_inches='tight')
plt.show()








