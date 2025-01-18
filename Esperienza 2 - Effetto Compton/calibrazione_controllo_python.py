import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit

# da fit_sorgenti_controllo.py
y=np.loadtxt('dati/ch_cal_21marzo_iniziofinemedia.txt', unpack=True)
dy=np.loadtxt('dati/dch_cal_21marzo_iniziofinemedia.txt', unpack=True)

x=np.array([1.173, 1.333, 0.511, 1.275, 0.662])

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
# Compute chi-square
residuals1 = (y - retta(x, *popt_retta)) / dy
chi_square1 = np.sum(residuals1 ** 2)

# Compute chi-square
residuals2 = (y - parabola(x, *popt_parabola)) / dy
chi_square2 = np.sum(residuals2 ** 2)

# Print chi-square
print("Chi-square retta:", chi_square1)
print("Chi-square parabola:", chi_square2)
plt.plot(x_range, parabola(x_range, *popt_parabola), 'r-', label='Parabola fit: a=%5.3f ± %5.3f, b=%5.3f ± %5.3f, c=%5.3f ± %5.3f' % (*popt_parabola, *perr_parabola))
plt.plot(x_range, retta(x_range, *popt_retta), 'g--', label='Retta fit: m=%5.3f ± %5.3f, q=%5.3f ± %5.3f' % (*popt_retta, *perr_retta))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
