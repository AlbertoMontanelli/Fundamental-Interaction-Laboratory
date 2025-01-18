import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# Definisci la funzione gaussiana
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Dati reali
rates = np.array([7,10,33,67,141,279,322,334,324,279,183,26,13,11])
angles = np.array([80, 85, 87, 88, 89, 91, 92, 93, 94, 95, 96, 100, 102, 105])
x = np.linspace(80, 105, 1000)

# Esegui il fit dei dati
initial_guess = [330, 92, 9]  # Valori iniziali per l'amplitude, la media e la deviazione standard
params, cov_matrix = curve_fit(gaussian, angles, rates, p0=initial_guess, absolute_sigma=True)

# Stampa i parametri del fit
print("Parametri del fit:")
print("Amplitude:", params[0], '+/-', np.sqrt(cov_matrix[0, 0]))
print("Mean:", params[1], '+/-', np.sqrt(cov_matrix[1, 1]))
print("Stddev:", params[2], '+/-', np.sqrt(cov_matrix[2, 2]))

# Calcola i valori predetti dalla funzione di fit
y_pred = gaussian(angles, *params)

# Calcola il residuo
residuals = rates - y_pred

# Calcola il chi-quadro
chi_square = np.sum((residuals / np.sqrt(rates))**2)

# Calcola i gradi di libertà
ndof = len(rates) - len(params)

# Calcola il chi-quadro ridotto
reduced_chi_square = chi_square / ndof
print("Chi-quadro ridotto:", reduced_chi_square)

# Formattazione dei parametri per la legenda
label_fit = (
    f"Fit di Gaussiana\n"
    f"$\mu = {params[1]:.2f} \pm {np.sqrt(cov_matrix[1, 1]):.2f}$\n"
    f"$\chi^2/\mathrm{{ndof}} = {reduced_chi_square:.2f}$"
)

# Disegna i dati e il fit
plt.title(r'Rate al variare di $\theta$', fontdict={'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20})
plt.errorbar(angles, rates, yerr=np.sqrt(rates), xerr=np.sqrt(1/12), fmt='.', label="Dati")
line_fit, = plt.plot(x, gaussian(x, *params), color='red', label="Fit di Gaussiana")
plt.xlabel(r"$\theta$ [deg]", fontdict={'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20})
plt.ylabel("Rate [Hz]", fontdict={'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20})

# Modifica del font della label nella legenda
font = {'family': 'serif', 'weight': 'normal', 'size': 24}

# Crea un oggetto Line2D per la legenda personalizzata
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label="Fit function: $e^{-\\frac{(x-\mu)^2}{2\sigma^2}}$"),
    Line2D([0], [0], color='none', label=f"$\mu = ({params[1]:.2f} \pm {np.sqrt(cov_matrix[1, 1]):.2f})$deg"),
    Line2D([0], [0], color='none', label=f"$\chi^2/\mathrm{{ndof}} = {chi_square:.0f}/{ndof:.0f}$")
]

# Aggiunta della legenda con le proprietà del font e il colore delle etichette
plt.legend(handles=legend_elements, prop=font, labelcolor='black')

plt.grid()







# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/rate.pdf', dpi=1200, bbox_inches='tight')


plt.show()




