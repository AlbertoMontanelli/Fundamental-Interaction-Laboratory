import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
import math



# Array di angoli in radianti
l = np.linspace(7,50,1500)
d = 3

'''
def simplified_equation(E, a, theta, beta):
    numerator = E - (E**2 / a) * np.cos(theta + beta) - E + (E**2 / a) * np.cos(theta)
    denominator = (1 + (E / a) * (1 - np.cos(theta))) * (1 + (E / a) * (1 - np.cos(theta + beta)))

    result = numerator / denominator
    return result
'''
def simplified_equation(E, a, theta, beta):
	return (E/(1 + (E/a)*(1-np.cos(theta-beta))))-(E/(1 + (E/a)*(1-np.cos(theta+beta))))



# Esempio di utilizzo
E_val = 1.333  # Sostituisci con il valore reale di E
a_val = 0.511 # Sostituisci con il valore reale di a
theta_val = np.pi/12  # Sostituisci con il valore reale di theta

w_E_MeV=(simplified_equation(1.333, a_val, np.pi*19/180, np.arcsin(3/35)))
print(w_E_MeV)
sigma_MeV=(-5617+np.sqrt(5617**2-4*(-513)*(-210-171)))/(-513*2)
print(sigma_MeV*2.2)

# Calcola l'arcoseno per ciascun angolo
beta_val = np.arcsin(d/l)

# Converti gli arcoseni da radianti a gradi
angoli = np.degrees(beta_val)


plt.figure('larghezza fotopicchi [MeV]')
plt.plot(l ,simplified_equation(E_val, a_val, theta_val, beta_val))
plt.xlabel('distanza sorgente-PMT01 [cm]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.ylabel('larghezza fotopicchi [MeV]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.title('Larghezza fotopichi spettro Compton vs distanza PMT01-sorgente', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.grid()


# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi



plt.savefig('grafici/div_angolare.pdf', dpi=1200, bbox_inches='tight')

plt.show()




