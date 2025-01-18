import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
import math


# Array di angoli in radianti
l = np.linspace(1,50,150)
d = 6 

'''
def simplified_equation(E, a, theta, beta):
    numerator = E - (E**2 / a) * np.cos(theta + beta) - E + (E**2 / a) * np.cos(theta)
    denominator = (1 + (E / a) * (1 - np.cos(theta))) * (1 + (E / a) * (1 - np.cos(theta + beta)))

    result = numerator / denominator
    return result
'''
def simplified_equation(E, a, theta, beta):
	return (E/(1 + (E/a)*(1-np.cos(theta))))-(E/(1 + (E/a)*(1-np.cos(theta+beta))))

    

# Esempio di utilizzo
E_val = 1.33  # Sostituisci con il valore reale di E
a_val = 0.511 # Sostituisci con il valore reale di a
theta_val = np.pi/12  # Sostituisci con il valore reale di theta
#beta_val = 0.2  # Sostituisci con il valore reale di beta





# Calcola l'arcoseno per ciascun angolo
beta_val = np.arcsin(d/l)

# Converti gli arcoseni da radianti a gradi
angoli = np.degrees(beta_val)

plt.figure('div angolare')
plt.grid()
plt.plot(l,angoli)

plt.figure('larghezza picchi')
plt.grid()
plt.plot(l ,simplified_equation(E_val, a_val, theta_val, beta_val))


plt.show()



