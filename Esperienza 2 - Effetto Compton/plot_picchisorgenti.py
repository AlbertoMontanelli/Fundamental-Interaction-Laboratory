import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt

#**********************************************************************************

# PLOT DOVE SI VEDONO TUTTI PICCHI DELLE SORGENTI PER OGNI VOLTA CHE ABBIAMO PRESO DEI DATI PER LA CALIBRAZIONE

#*********************************************************************************


# Load data from files into NumPy arrays
ch_array = np.loadtxt('dati/ch_array_plot.txt', delimiter='\t')
dch_array = np.loadtxt('dati/dch_array_plot.txt', delimiter='\t')

# Lista delle etichette per le variabili
labels = ['Co peak 1', 'Co peak 2', 'Na peak 1', 'Na peak 2', 'Cs']

# Lista delle date
dates = ['27 feb', '28 feb', '29 feb', '5 mar', '6 mar', '7 mar', '12 mar init', '12 mar fin', '13 mar init', '13 mar fin']

# Creazione del grafico
for i, label in enumerate(labels):
    values = ch_array[:, i]
    errors = dch_array[:, i]
    plt.errorbar(dates, values, yerr=errors, fmt='o', label=label)

# Personalizzazione del grafico
plt.grid(linewidth=0.5)
plt.xlabel('Date')
plt.ylim(2450, 6750)
plt.ylabel('channels')
plt.title('Dipendenza calibrazioni nel tempo')
plt.legend()
plt.xticks(rotation=45)  # Ruota le etichette sull'asse x per una migliore leggibilità
plt.yticks(np.arange(2450, 6750, 100))  # Imposta gli intervalli sull'asse y
# Mostra il grafico
plt.tight_layout()  # Ottimizza la disposizione degli elementi nel grafico
plt.show()


