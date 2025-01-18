import numpy as np
import matplotlib.pyplot as plt


# Definisci i valori per theta, E1, E2 e a
theta_values = np.linspace(- np.pi, np.pi, 1000)
E1 = 1.17
E2 = 1.33
m_e = 0.511



def E_fin(E, theta, m):
    return E/(1 + ((E/m)*(1 - np.cos(theta))))

# Definisci la funzione
def my_function(theta, E1, E2, a):
    return (E2 - E1) / (1 + ((E1 + E2) / a) * (1 - np.cos(theta)) + ((E1 * E2) / a) * (1 - np.cos(theta))**2)



# Calcola i valori della funzione per i parametri dati
y_values =-E_fin(E1, theta_values, m_e)+E_fin(E2, theta_values, m_e)

plt.figure('1')
# Plotta i risultati
#plt.errorbar(th,deltaE, dE, fmt ='o')
plt.plot(theta_values, y_values, )
plt.xlabel('$\\theta$ [rad]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.ylabel('$\Delta$E [MeV]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})
plt.grid()
plt.title('Differenza in energia tra i due fotopicchi dello spettro Compton al variare di $\\theta$', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':26,})




# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/Delta_E.pdf', dpi=1200, bbox_inches='tight')

plt.show()
