import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.interpolate import interp1d

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }


def diff(d):
	return d-(280-d)
	
	
# nei file ho media fit, err media fit, sigma fit, err sigma fit, mediana, err mediana, medie troncate, err medie troncate	
a=np.loadtxt("04-18/medie_1e2_2.5cm.txt", unpack=True)
b=np.loadtxt("04-23/medie_1e2_100cm.txt", unpack=True)
c=np.loadtxt("04-16/medie_1e2_140cm.txt", unpack=True)
d=np.loadtxt("04-17/medie_1e2_220cm.txt", unpack=True)
e=np.loadtxt("04-19/medie_1e2_262.5cm.txt", unpack=True)
f=np.loadtxt("05-02/medie_1e2_180cm.txt", unpack=True)


#per la calibrazione
dist=np.array([2.5, 100, 140, 220, 262.5, 180], dtype=float)

#per le medie dei fit
#t=np.array([ a[0], b[0], c[0], d[0], e[0], f[0]  ]) 
#dt=np.array([ a[1], b[1], c[1], d[1], e[1], f[1]  ])

#per le mediane
#t=np.array([ a[4], b[4], c[4], d[4], e[4], f[4]  ])
#dt=np.array([ a[5], b[5], c[5], d[5], e[5], f[5]  ])

#per le medie troncate
t=np.array([ a[6], b[6], c[6], d[6], e[6], f[6]  ]) 
dt=np.array([ a[7], b[7], c[7], d[7], e[7], f[7]  ])

dd=np.full(len(dist), 1)

x = np.linspace(min(dist), max(dist), 10000)  # Definizione di x per il grafico



def retta(x, m, q):
    return m * x + q


popt, pcov = curve_fit(retta, dist, t, sigma=dt)


for i in range(10):
	dtot=np.sqrt(dt**2+(dd*popt[0])**2)
	popt, pcov = curve_fit(retta, dist, t, sigma=dtot)
	chisq=(((t-retta(dist, *popt))/dtot)**2).sum()
	

m, q = popt
m_err, q_err = np.sqrt(np.diag(pcov))
degrees_of_freedom = len(dist) - len(popt)
print(chisq/degrees_of_freedom)

#***************************************************************************************************************************************************************************************************

# GRAFICO

#***************************************************************************************************************************************************************************************************

plt.plot(x, retta(x, m, q))

plt.errorbar(dist, t, None, dd ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
 
plt.xlabel('$x_1$ PMT03 [cm]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })
plt.ylabel('$\Delta$T PMT01-PMT02 [ns]', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })
plt.grid()

legenda = "Parametri del fit:\n"
legenda += "$m_{{cal}}$ = ({:.2f} $\pm$ {:.2f})$\cdot$ 10$^{{-2}}$ ns/cm\n".format(m*100, m_err*100)
legenda += "$q_{{cal}}$ = ({:.2f} $\pm$ {:.2f}) ns\n".format(q, q_err)
legenda += "$1/m_{{cal}}$ = ({:.2f} $\pm$ {:.2f}) cm/ns\n".format((1/m), (m_err*(1/m**2)))
legenda += "$\chi^2$ / ndof = {:.0f} / {:d}".format(chisq,degrees_of_freedom)
plt.text(0.1, 0.9, legenda, transform=plt.gca().transAxes, fontsize=28, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Fit per la calibrazione spazio-tempo', fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 26, })



cal_1e2=np.array([m, m_err, q, q_err, pcov[0][1] ])
#np.savetxt('cal_1e2.txt', cal_1e2)

# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi

plt.savefig('grafici/fit_cal.pdf', dpi=1200, bbox_inches='tight')



plt.show()




