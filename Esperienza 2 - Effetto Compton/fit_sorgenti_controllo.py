import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *
from convertitore import convertitore
from fit_sorgenti import fit_cobalto, fit_sodio, fit_cesio
from fit_sorgenti2 import fit_cobalto2, fit_sodio2, fit_cesio2
from calibrazione import calibrazione
from compton import compton_fit
from compton2 import compton_fit2


#************************************************************************************************************************************************************************************************

# CALCOLO ERRORI ANDAMENTI SORGENTI E COMPTON PER VARIAZIONE RANGE, BINNING E BKG FUNCTION

#*************************************************************************************************************************************************************************************************

picco_Co1, err_picco_Co1, sigma_Co1, err_sigma_Co1, picco_Co2, err_picco_Co2, sigma_Co2, err_sigma_Co2 = plb.loadtxt('dati/andamento_cobalto.txt', unpack=True)
picco_Na1, err_picco_Na1, sigma_Na1, err_sigma_Na1, picco_Na2, err_picco_Na2, sigma_Na2, err_sigma_Na2 = plb.loadtxt('dati/andamento_sodio.txt', unpack=True)
picco_Cs, err_picco_Cs, sigma_Cs, err_sigma_Cs = plb.loadtxt('dati/andamento_cesio.txt', unpack=True)
picco_1, err_picco_1, sigma_1, err_sigma_1, picco_2, err_picco_2, sigma_2, err_sigma_2 = plb.loadtxt('dati/andamento_compton.txt', unpack=True)

media_Co1=np.mean(picco_Co1)
err_media_Co1=np.mean(err_picco_Co1)
media_Co2=np.mean(picco_Co2)
err_media_Co2=np.mean(err_picco_Co2)

media_Na1=np.mean(picco_Na1)
err_media_Na1=np.mean(err_picco_Na1)
media_Na2=np.mean(picco_Na2)
err_media_Na2=np.mean(err_picco_Na2)

media_Cs=np.mean(picco_Cs)
err_media_Cs=np.mean(err_picco_Cs)

media_1=np.mean(picco_1)
err_media_1=np.mean(err_picco_1)
media_2=np.mean(picco_2)
err_media_2=np.mean(err_picco_2)

err_Co1=(max(picco_Co1)-min(picco_Co1))/2 
err_Co2=(max(picco_Co2)-min(picco_Co2))/2 

err_Na1= (max(picco_Na1)-min(picco_Na1))/2 
err_Na2=(max(picco_Na2)-min(picco_Na2))/2 

err_Cs= (max(picco_Cs)-min(picco_Cs))/2 

err_1=(max(picco_1)-min(picco_1))/2 
err_2= (max(picco_2)-min(picco_2))/2 

dCo1=(err_Co1/media_Co1)
dCo2=(err_Co2/media_Co2)

dNa1=(err_Na1/media_Na1)
dNa2=(err_Na2/media_Na2)

dCs=(err_Cs/media_Cs)

dCompton1=(err_1/media_1)
dCompton2=(err_2/media_2)

"""
print(media_Co1,err_Co1)
print(media_Co2, err_Co2)
print(media_Na1, err_Na1)
print(media_Na2, err_Na2)
print(media_Cs, err_Cs)
print(media_1, err_1)
print(media_2, err_2)
"""
#***************************************************

# DATI VARIE SORGENTI

#***************************************************

cobalto1= plb.loadtxt('02-27/calibrazione/cobalto.dat',unpack=True)
cobalto2= plb.loadtxt('02-28/calibrazione/cobalto.dat',unpack=True)
cobalto3= plb.loadtxt('02-29/calibrazione/cobalto.dat',unpack=True)
cobalto4= plb.loadtxt('03-05/calibrazione/cobalto.dat',unpack=True)
cobalto5= plb.loadtxt('03-06/calibrazione/cobalto.dat',unpack=True)
cobalto6=convertitore("03-07/calibrazione/cobalto_esadecimale.dat")
cobalto7=convertitore("03-12/calibrazione/inizio/cobalto_esadecimale.dat")
cobalto8=convertitore("03-12/calibrazione/fine/cobalto_esadecimale.dat")
cobalto9=convertitore("03-13/calibrazione/inizio/cobalto_esadecimale.dat")
cobalto10=convertitore("03-13/calibrazione/fine/cobalto_esadecimale.dat")
cobalto11=convertitore("03-19/calibrazione/inizio/cobalto_inizio_esadecimale.dat")
cobalto12=convertitore("03-19/calibrazione/fine/cobalto_fine_esadecimale.dat")
cobalto13=convertitore("03-20/calibrazione/cobalto_esadecimale.dat")
cobalto14=convertitore("03-21/calibrazione/inizio/cobalto_esadecimale.dat")
cobalto15=convertitore("03-21/calibrazione/metà/cobalto_esadecimale.dat")
cobalto16=convertitore("03-21/calibrazione/fine/cobalto_esadecimale.dat")

sodio1 = plb.loadtxt('02-27/calibrazione/sodio.dat', unpack=True)
sodio2 = plb.loadtxt('02-28/calibrazione/sodio.dat', unpack=True)
sodio3 = plb.loadtxt('02-29/calibrazione/sodio.dat', unpack=True)
sodio4 = plb.loadtxt('03-05/calibrazione/sodio.dat', unpack=True)
sodio5 = plb.loadtxt('03-06/calibrazione/sodio.dat', unpack=True)
sodio6 = convertitore("03-07/calibrazione/sodio_esadecimale.dat")
sodio7 = convertitore("03-12/calibrazione/inizio/sodio_esadecimale.dat")
sodio8 = convertitore("03-12/calibrazione/fine/sodio_esadecimale.dat")
sodio9 = convertitore("03-13/calibrazione/inizio/sodio_esadecimale.dat")
sodio10 = convertitore("03-13/calibrazione/fine/sodio_esadecimale.dat")
sodio11=convertitore("03-19/calibrazione/inizio/sodio_inizio_esadecimale.dat")
sodio12=convertitore("03-19/calibrazione/fine/sodio_fine_esadecimale.dat")
sodio13=convertitore("03-20/calibrazione/sodio_esadecimale.dat")
sodio14=convertitore("03-21/calibrazione/inizio/sodio_esadecimale.dat")
sodio15=convertitore("03-21/calibrazione/metà/sodio_esadecimale.dat")
sodio16=convertitore("03-21/calibrazione/fine/sodio_esadecimale.dat")

cesio1 = plb.loadtxt('02-27/calibrazione/cesio.dat', unpack=True)
cesio2 = plb.loadtxt('02-28/calibrazione/cesio.dat', unpack=True)
cesio3 = plb.loadtxt('02-29/calibrazione/cesio.dat', unpack=True)
cesio4 = plb.loadtxt('03-05/calibrazione/cesio.dat', unpack=True)
cesio5 = plb.loadtxt('03-06/calibrazione/cesio.dat', unpack=True)
cesio6 = convertitore("03-07/calibrazione/cesio_esadecimale.dat")
cesio7 = convertitore("03-12/calibrazione/inizio/cesio_esadecimale.dat")
cesio8 = convertitore("03-12/calibrazione/fine/cesio_esadecimale.dat")
cesio9 = convertitore("03-13/calibrazione/inizio/cesio_esadecimale.dat")
cesio10 = convertitore("03-13/calibrazione/fine/cesio_esadecimale.dat")
cesio11=convertitore("03-19/calibrazione/inizio/cesio_inizio_esadecimale.dat")
cesio12=convertitore("03-19/calibrazione/fine/cesio_fine_esadecimale.dat")
cesio13=convertitore("03-20/calibrazione/cesio_esadecimale.dat")
cesio14=convertitore("03-21/calibrazione/inizio/cesio_esadecimale.dat")
cesio15=convertitore("03-21/calibrazione/metà/cesio_esadecimale.dat")
cesio16=convertitore("03-21/calibrazione/fine/cesio_esadecimale.dat")


#*****************************************************************

# FIT PER LE VARIE SORGENTI

#*****************************************************************
"""
a_Co_1, a_dCo_1, b_Co_1, b_dCo_1=fit_cobalto2(cobalto1)
a_Na_1, a_dNa_1, b_Na_1, b_dNa_1=fit_sodio2(sodio1)
Cs_1, dCs_1=fit_cesio2(cesio1)

a_Co_2, a_dCo_2, b_Co_2, b_dCo_2=fit_cobalto2(cobalto2)
a_Na_2, a_dNa_2, b_Na_2, b_dNa_2=fit_sodio2(sodio2)
Cs_2, dCs_2=fit_cesio2(cesio2)

a_Co_3, a_dCo_3, b_Co_3, b_dCo_3=fit_cobalto2(cobalto3)
a_Na_3, a_dNa_3, b_Na_3, b_dNa_3=fit_sodio2(sodio3)
Cs_3, dCs_3=fit_cesio2(cesio3)

a_Co_4, a_dCo_4, b_Co_4, b_dCo_4=fit_cobalto2(cobalto4)
a_Na_4, a_dNa_4, b_Na_4, b_dNa_4=fit_sodio2(sodio4)
Cs_4, dCs_4=fit_cesio2(cesio4)

a_Co_5, a_dCo_5, b_Co_5, b_dCo_5=fit_cobalto2(cobalto5)
a_Na_5, a_dNa_5, b_Na_5, b_dNa_5=fit_sodio2(sodio5)
Cs_5, dCs_5=fit_cesio2(cesio5)

a_Co_6, a_dCo_6, b_Co_6, b_dCo_6=fit_cobalto(cobalto6, 500)
a_Na_6, a_dNa_6, b_Na_6, b_dNa_6=fit_sodio(sodio6, 500)
Cs_6, dCs_6=fit_cesio(cesio6, 500)

a_Co_7, a_dCo_7, b_Co_7, b_dCo_7=fit_cobalto(cobalto7)
a_Na_7, a_dNa_7, b_Na_7, b_dNa_7=fit_sodio(sodio7)
Cs_7, dCs_7=fit_cesio(cesio7)

a_Co_8, a_dCo_8, b_Co_8, b_dCo_8=fit_cobalto(cobalto8, 500)
a_Na_8, a_dNa_8, b_Na_8, b_dNa_8=fit_sodio(sodio8, 500)
Cs_8, dCs_8=fit_cesio(cesio8, 500)

a_Co_9, a_dCo_9, b_Co_9, b_dCo_9=fit_cobalto(cobalto9, 500)
a_Na_9, a_dNa_9, b_Na_9, b_dNa_9=fit_sodio(sodio9, 500)
Cs_9, dCs_9=fit_cesio(cesio9, 500)

a_Co_10, a_dCo_10, b_Co_10, b_dCo_10=fit_cobalto(cobalto10, 500)
a_Na_10, a_dNa_10, b_Na_10, b_dNa_10=fit_sodio(sodio10, 500)
Cs_10, dCs_10=fit_cesio(cesio10, 500)
"""
a_Co_11, a_dCo_11, b_Co_11, b_dCo_11 = fit_cobalto(cobalto11, 400)
a_Na_11, a_dNa_11, b_Na_11, b_dNa_11 = fit_sodio(sodio11, 400)
Cs_11, dCs_11 = fit_cesio(cesio11, 500)
"""
a_Co_12, a_dCo_12, b_Co_12, b_dCo_12 = fit_cobalto(cobalto12, 400)
a_Na_12, a_dNa_12, b_Na_12, b_dNa_12 = fit_sodio(sodio12, 400)
Cs_12, dCs_12 = fit_cesio(cesio12, 500)


a_Co_13, a_dCo_13, b_Co_13, b_dCo_13 = fit_cobalto(cobalto13, 400)
a_Na_13, a_dNa_13, b_Na_13, b_dNa_13 = fit_sodio(sodio13, 400)
Cs_13, dCs_13 = fit_cesio(cesio13, 400)

a_Co_14, a_dCo_14, b_Co_14, b_dCo_14 = fit_cobalto(cobalto14, 350)
a_Na_14, a_dNa_14, b_Na_14, b_dNa_14 = fit_sodio(sodio14, 350)
Cs_14, dCs_14 = fit_cesio(cesio14, 350)

a_Co_15, a_dCo_15, b_Co_15, b_dCo_15 = fit_cobalto(cobalto15, 350)
a_Na_15, a_dNa_15, b_Na_15, b_dNa_15 = fit_sodio(sodio15, 350)
Cs_15, dCs_15 = fit_cesio(cesio15, 350)

a_Co_16, a_dCo_16, b_Co_16, b_dCo_16 = fit_cobalto(cobalto16, 350)
a_Na_16, a_dNa_16, b_Na_16, b_dNa_16 = fit_sodio(sodio16, 350)
Cs_16, dCs_16 = fit_cesio(cesio16, 350)
"""

input()

"""
# Inizializzazione delle liste vuote per i valori ch e dch
ch_list = []
dch_list = []


# Ciclo for per creare ch e dch per ciascun campione
for i in range(2, 4):
    ch = np.array([globals()[f"a_Co_{i}"], globals()[f"b_Co_{i}"], globals()[f"a_Na_{i}"], globals()[f"b_Na_{i}"],globals()[f"Cs_{i}"]])
    dch = np.array([globals()[f"a_dCo_{i}"], globals()[f"b_dCo_{i}"], globals()[f"a_dNa_{i}"], globals()[f"b_dNa_{i}"], globals()[f"dCs_{i}"]])
    # Aggiungere i valori ch e dch alle rispettive liste
    ch_list.append(ch)
    dch_list.append(dch)

#**********************************************************************************************************************************

# ROBA CHE SERVE PER plot_picchisorgenti.py

# Salvataggio di ch_array, 
#np.savetxt('dati/ch_array_plot.txt', ch_list,  delimiter='\t', header='a_Co\tb_Co\ta_Na\tb_Na\tCs', comments='')

# Salvataggio di dch_array
#np.savetxt('dati/dch_array_plot.txt', dch_list, fmt='%f', delimiter='\t', header='a_dCo\tb_dCo\ta_dNa\tb_dNa\tdCs', comments='')

#*******************************************************************************************************************************


ch_array=np.array(ch_list)
dch_array=np.array(dch_list)

media_a_Co=np.mean(ch_array[:, 0])
media_b_Co=np.mean(ch_array[:, 1])
media_a_Na=np.mean(ch_array[:, 2])
media_b_Na=np.mean(ch_array[:, 3])
media_Cs=np.mean(ch_array[:, 4])

media_err_a_Co=np.sqrt(np.sum(dch_array[:, 0]**2)/(len(dch_array[:, 0])**2))
media_err_b_Co=np.sqrt(np.sum(dch_array[:, 1]**2)/(len(dch_array[:, 1])**2))
media_err_a_Na=np.sqrt(np.sum(dch_array[:, 2]**2)/(len(dch_array[:, 2])**2))
media_err_b_Na=np.sqrt(np.sum(dch_array[:, 3]**2)/(len(dch_array[:, 3])**2))
media_err_Cs=np.sqrt(np.sum(dch_array[:, 4]**2)/(len(dch_array[:, 4])**2))


dch_array[:, 0]=np.sqrt((ch_array[:, 0]*dCo1)**2+dch_array[:,0]**2)
dch_array[:, 1]=np.sqrt((ch_array[:, 1]*dCo2)**2+dch_array[:,1]**2)
dch_array[:, 2]=np.sqrt((ch_array[:, 2]*dNa1)**2+dch_array[:,2]**2)
dch_array[:, 3]=np.sqrt((ch_array[:, 3]*dNa2)**2+dch_array[:,3]**2)
dch_array[:, 4]=np.sqrt((ch_array[:, 4]*dCs)**2+dch_array[:,4]**2)

err_a_Co=np.sqrt( ((np.max(ch_array[:, 0])-np.min(ch_array[:,0]))/2)**2+media_err_a_Co**2+(dCo1*media_a_Co)**2)
err_b_Co=np.sqrt( ((np.max(ch_array[:, 1])-np.min(ch_array[:,1]))/2)**2+media_err_b_Co**2+(dCo2*media_b_Co)**2)
err_a_Na=np.sqrt( ((np.max(ch_array[:, 2])-np.min(ch_array[:,2]))/2)**2+media_err_a_Na**2+(dNa1*media_a_Na)**2)
err_b_Na=np.sqrt( ((np.max(ch_array[:, 3])-np.min(ch_array[:,3]))/2)**2+media_err_b_Na**2+(dNa2*media_b_Na)**2)
err_Cs=np.sqrt( ((np.max(ch_array[:, 4])-np.min(ch_array[:,4]))/2)**2+media_err_Cs**2+(dCs*media_Cs)**2)

print('Co1= %f +- %f' %(media_a_Co, err_a_Co))
print('Co2= %f +- %f' %(media_b_Co, err_b_Co))
print('Na1= %f +- %f' %(media_a_Na, err_a_Na))
print('Na2= %f +- %f' %(media_b_Na, err_b_Na))
print('Cs= %f +- %f' %(media_Cs, err_Cs))


ch_cal=np.array([media_a_Co, media_b_Co, media_a_Na,media_b_Na, media_Cs])
dch_cal=np.array([err_a_Co, err_b_Co, err_a_Na, err_b_Na, err_Cs])

np.savetxt('dati/ch_cal_28e29febbraio_iniziofinemedia.txt', ch_cal)
np.savetxt('dati/dch_cal_28e29febbraio_iniziofinemedia.txt', dch_cal)

np.savetxt('dati/ch_cal_28e29febbraio_iniziofine.txt', ch_array)
np.savetxt('dati/dch_cal_28e29febbraio_iniziofine.txt', dch_array)
"""


