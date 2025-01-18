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

#****************************************************************************

# fit spettri sorgenti

#***************************************************************************


cobalto=convertitore("7 marzo/calibrazione/cobalto_esadecimale.dat")
cobalto2= plb.loadtxt('27 febbraio/calibrazione/cobalto.dat',unpack=True)
cobalto3= plb.loadtxt('28 febbraio/calibrazione/cobalto.dat',unpack=True)
cobalto4= plb.loadtxt('29 febbraio/calibrazione/cobalto.dat',unpack=True)
#cobalto5= plb.loadtxt('5 marzo/calibrazione/cobalto.dat',unpack=True)
cobalto6= plb.loadtxt('6 marzo/calibrazione/cobalto.dat',unpack=True)


sodio=convertitore("7 marzo/calibrazione/sodio_esadecimale.dat")
sodio2= plb.loadtxt('27 febbraio/calibrazione/sodio.dat',unpack=True)
sodio3= plb.loadtxt('28 febbraio/calibrazione/sodio.dat',unpack=True)
sodio4= plb.loadtxt('29 febbraio/calibrazione/sodio.dat',unpack=True)
#sodio5= plb.loadtxt('5 marzo/calibrazione/sodio.dat',unpack=True)
sodio6= plb.loadtxt('6 marzo/calibrazione/sodio.dat',unpack=True)

cesio=convertitore("7 marzo/calibrazione/cesio_esadecimale.dat")
cesio2= plb.loadtxt('27 febbraio/calibrazione/cesio.dat',unpack=True)
cesio3= plb.loadtxt('28 febbraio/calibrazione/cesio.dat',unpack=True)
cesio4= plb.loadtxt('29 febbraio/calibrazione/cesio.dat',unpack=True)
#cesio5= plb.loadtxt('5 marzo/calibrazione/cesio.dat',unpack=True)
cesio6= plb.loadtxt('6 marzo/calibrazione/cesio.dat',unpack=True)

a_Co, a_dCo, b_Co, b_dCo=fit_cobalto(cobalto)
a_Na, a_dNa, b_Na, b_dNa=fit_sodio(sodio)
Cs, dCs=fit_cesio(cesio)

a_Co_2, a_dCo_2, b_Co_2, b_dCo_2=fit_cobalto2(cobalto2)
a_Na_2, a_dNa_2, b_Na_2, b_dNa_2=fit_sodio2(sodio2)
Cs_2, dCs_2=fit_cesio2(cesio2)

a_Co_3, a_dCo_3, b_Co_3, b_dCo_3=fit_cobalto2(cobalto3)
a_Na_3, a_dNa_3, b_Na_3, b_dNa_3=fit_sodio2(sodio3)
Cs_3, dCs_3=fit_cesio2(cesio3)

a_Co_4, a_dCo_4, b_Co_4, b_dCo_4=fit_cobalto2(cobalto4)
a_Na_4, a_dNa_4, b_Na_4, b_dNa_4=fit_sodio2(sodio4)
Cs_4, dCs_4=fit_cesio2(cesio4)

#a_Co_5, a_dCo_5, b_Co_5, b_dCo_5=fit_cobalto2(cobalto5)
#a_Na_5, a_dNa_5, b_Na_5, b_dNa_5=fit_sodio2(sodio5)
#Cs_5, dCs_5=fit_cesio2(cesio5)

a_Co_6, a_dCo_6, b_Co_6, b_dCo_6=fit_cobalto2(cobalto6)
a_Na_6, a_dNa_6, b_Na_6, b_dNa_6=fit_sodio2(sodio6)
Cs_6, dCs_6=fit_cesio2(cesio6)

#*********************************************************************************************

#calibrazioni

#********************************************************************************************

ch=np.array([a_Co, b_Co, a_Na, b_Na, Cs])
dch=np.array([a_dCo, b_dCo, a_dNa, b_dNa, dCs])

ch2=np.array([a_Co_2, b_Co_2, a_Na_2, b_Na_2, Cs_2])
dch2=np.array([a_dCo_2, b_dCo_2, a_dNa_2, b_dNa_2, dCs_2])

ch3=np.array([a_Co_3, b_Co_3, a_Na_3, b_Na_3, Cs_3])
dch3=np.array([a_dCo_3, b_dCo_3, a_dNa_3, b_dNa_3, dCs_3])

ch4=np.array([a_Co_4, b_Co_4, a_Na_4, b_Na_4, Cs_4])
dch4=np.array([a_dCo_4, b_dCo_4, a_dNa_4, b_dNa_4, dCs_4])

#ch5=np.array([a_Co_5, b_Co_5, a_Na_5, b_Na_5, Cs_5])
#dch5=np.array([a_dCo_5, b_dCo_5, a_dNa_5, b_dNa_5, dCs_5])

ch6=np.array([a_Co_6, b_Co_6, a_Na_6, b_Na_6, Cs_6])
dch6=np.array([a_dCo_6, b_dCo_6, a_dNa_6, b_dNa_6, dCs_6])


m, m_err, q, q_err, c, c_err, b, b_err, a, a_err=calibrazione(ch, dch)

m2, m_err2, q2, q_err2, c2, c_err2, b2, b_err2, a2, a_err2=calibrazione(ch2, dch2)

m3, m_err3, q3, q_err3, c3, c_err3, b3, b_err3, a3, a_err3=calibrazione(ch3, dch3)

m4, m_err4, q4, q_err4, c4, c_err4, b4, b_err4, a4, a_err4=calibrazione(ch4, dch4)

#m5, m_err5, q5, q_err5, c5, c_err5, b5, b_err5, a5, a_err5=calibrazione(ch5, dch5)

m6, m_err6, q6, q_err6, c6, c_err6, b6, b_err6, a6, a_err6=calibrazione(ch6, dch6)

m_array=np.array([m, m2, m3, m4,  m6])
m_err_array=np.array([m_err, m_err2, m_err3, m_err4,  m_err6])

q_array=np.array([q, q2, q3, q4,  q6])
q_err_array=np.array([q_err, q_err2, q_err3, q_err4, q_err6])

c_array=np.array([c, c2, c3, c4, c6])
c_err_array=np.array([c_err, c_err2, c_err3, c_err4, c_err6])

b_array=np.array([b, b2, b3, b4, b6])
b_err_array=np.array([b_err, b_err2, b_err3, b_err4, b_err6])

a_array=np.array([a, a2, a3, a4, a6])
a_err_array=np.array([a_err, a_err2, a_err3, a_err4, a_err6])


np.savetxt("m.txt", m_array)
np.savetxt("m_err.txt", m_err_array)
np.savetxt("q.txt", q_array)
np.savetxt("q_err.txt", q_err_array)
"""
np.savetxt("c.txt", c_array)
np.savetxt("c_err.txt", c_err_array)
np.savetxt("b.txt", b_array)
np.savetxt("b_err.txt", b_err_array)
np.savetxt("a.txt", a_array)
np.savetxt("a_err.txt", a_err_array)
"""
"""
m_media=np.mean(m_array)
dm_media=(m_array-m_media)**2
dm_media=np.sqrt( (np.sum(dm_media))/(len(m_array)-1))

q_media=np.mean(q_array)
dq_media=(q_array-q_media)**2
dq_media=np.sqrt( (np.sum(dq_media))/(len(q_array)-1))

a_media=np.mean(a_array)
da_media=(a_array-a_media)**2
da_media=np.sqrt( (np.sum(da_media))/(len(a_array)-1))

b_media=np.mean(b_array)
db_media=(b_array-b_media)**2
db_media=np.sqrt( (np.sum(db_media))/(len(b_array)-1))

c_media=np.mean(c_array)
c_media=(c_array-c_media)**2
c_media=np.sqrt( (np.sum(c_media))/(len(c_array)-1))
"""

#***************************************************************************

# SPETTRO COMPTON

#***************************************************************************
"""
compton = plb.loadtxt("5 marzo/compton/compton_15deg_35cm.dat", unpack=True)
# fit function double gauss+polynomial
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4000, 6400)
fit_function_compton.SetParameters(726, 5200, 217, 672, 5823, 192, 2914, -0.6785, 3.58e-5) 
a_compton, a_err_compton, b_compton, b_err_compton=compton2(compton, fit_function_compton)
print(a_compton, a_err_compton)
print(b_compton, b_err_compton)
"""
compton2= plb.loadtxt("28 febbraio/compton/compton_73deg_28feb.dat", unpack=True)
# fit function double gauss+polynomial
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4000, 6400)
fit_function_compton.SetParameters(726, 5200, 217, 672, 5823, 192, 2914, -0.6785, 3.58e-5) 
a_compton2, a_err_compton2, b_compton2, b_err_compton2=compton_fit2(compton2, fit_function_compton)

"""
compton3=convertitore("compton_18deg_35cm_esadecimale.dat")
# fit function double gauss+polynomial
fit_function_compton= TF1("fit_function_compton", "gaus(0)+gaus(3)+pol2(6)", 4000, 6400)
fit_function_compton.SetParameters(726, 5200, 217, 672, 5823, 192, 2914, -0.6785, 3.58e-5) 
a_compton3, a_err_compton3, b_compton3, b_err_compton3=compton(compton3, fit_function_compton)
"""

#********************************************************************************

#CONVERSIONE PICCHI COMPTON IN ENERGIA

#***********************************************************************************
m_media=(m3+m4)/2
q_media=(q3+q4)/2
m_err_media = np.sqrt((1/2 * m_err3)**2 + (1/2 * m_err4)**2)
q_err_media = np.sqrt((1/2 * q_err3)**2 + (1/2 * q_err4)**2)

a_media=(a3+a4)/2
b_media=(b3+b4)/2
c_media=(c3+c4)/2

a_err_media=np.sqrt((1/2 * a_err3)**2 + (1/2 * a_err4)**2 )
b_err_media=np.sqrt((1/2 * b_err3)**2 + (1/2 * b_err4)**2 )
c_err_media=np.sqrt((1/2 * c_err3)**2 + (1/2 * c_err4)**2 )

def conversionelineare(ch, q, m):
	E_lin=(ch-q)/m
	return E_lin

E_lin1=conversionelineare(5000, q3, m3)
E_lin2=conversionelineare(5000, q4, m4)
print(E_lin1)
print(E_lin2)	
def E_lin_err(m, q, m_err, q_err, ch, ch_err):
    # Calcolo delle derivate parziali rispetto ai parametri
    dE_dm = -1 / (m**2)
    dE_dq = -1 / m
    dE_dch = 1 / m

    # Calcolo dell'errore utilizzando le derivate parziali
    E_lin_err = np.sqrt((dE_dm * m_err)**2 + (dE_dq * q_err)**2 + (dE_dch * ch_err)**2)
    
    return E_lin_err
	
def conversionepolinomiale(ch, a, b, c):
	E_pol=(-b+np.sqrt(b**2-4*a*(c-ch)))/(2*a)
	return E_pol

def E_pol_err(ch, ch_err, a, b, c, a_err, b_err, c_err):
	# Calcolo delle derivate parziali rispetto ai parametri
	dE_da = (1 / (2*a)) * ((-b + np.sqrt(b**2 - 4*a*(c - ch))) / (2*np.sqrt(b**2 - 4*a*(c - ch))))
	dE_db = (1 / (2*a)) * (1 / np.sqrt(b**2 - 4*a*(c - ch)))
	dE_dc = (1 / (2*a)) * (-1 / np.sqrt(b**2 - 4*a*(c - ch)))
	dE_dch = (1 / (2*a)) * (1 / np.sqrt(b**2 - 4*a*(c - ch)))

	# Calcolo dell'errore utilizzando le derivate parziali
	E_pol_err = np.sqrt((dE_da * a_err)**2 + (dE_db * b_err)**2 + (dE_dc * c_err)**2 + (dE_dch * ch_err)**2)
    
	return E_pol_err
	


		
E_a_lin=conversionelineare(a_compton2, q_media, m_media)
E_b_lin=conversionelineare(b_compton2, q_media, m_media)
dE_a_lin=E_lin_err(m_media, q_media, m_err_media, q_err_media, a_compton2, a_err_compton2)
dE_b_lin=E_lin_err(m_media, q_media, m_err_media, q_err_media, b_compton2, b_err_compton2)

E_a_pol=conversionepolinomiale(a_compton2, a_media, b_media, c_media)
E_b_pol=conversionepolinomiale(b_compton2, a_media, b_media, c_media)
dE_a_pol=E_pol_err(a_compton2, a_err_compton2, a_media, b_media, c_media, a_err_media, b_err_media, c_err_media)
dE_b_pol=E_pol_err(b_compton2, b_err_compton2, a_media, b_media, c_media, a_err_media, b_err_media, c_err_media)

print('\n')
print(E_a_lin)
print(E_b_lin)
print(E_a_pol)
print(E_b_pol)
print('\n')

def m_E(E, E_in, theta):
	return (E*E_in*(1-np.cos(theta)))/(E_in-E)
	
def m_E_err(E, E_in, E_err, theta,  theta_err):
    
	d_m_E_dE = (E_in * (1 - np.cos(theta))) / (E_in - E) + (E * E_in * (1 - np.cos(theta))) / ((E_in - E) ** 2)
	d_m_E_dtheta = (E * E_in * np.sin(theta) * (E - E_in + E * np.cos(theta))) / ((E_in - E) ** 2)
	m_E_err = np.sqrt((d_m_E_dE * E_err) ** 2 + (d_m_E_dtheta * theta_err) ** 2)

	return m_E_err
	

	
m_e1a_lin=m_E(E_a_lin, 1.173, 20*np.pi/180)
m_e1b_lin=m_E(E_b_lin, 1.333, 20*np.pi/180)
dm_e1a_lin=m_E_err(m_e1a_lin, 1.173, dE_a_lin, 20*np.pi/180, np.pi/180)
dm_e1b_lin=m_E_err(m_e1b_lin, 1.333, dE_b_lin, 20*np.pi/180, np.pi/180)

m_e1a_pol=m_E(E_a_pol, 1.173, 20*np.pi/180)
m_e1b_pol=m_E(E_b_pol, 1.333, 20*np.pi/180)
dm_e1a_pol=m_E_err(m_e1a_pol, 1.173, dE_a_pol, 20*np.pi/180, np.pi/180)
dm_e1b_pol=m_E_err(m_e1b_pol, 1.333, dE_b_pol, 20*np.pi/180, np.pi/180)

print(m_e1a_lin, dm_e1a_lin )
print(m_e1b_lin, dm_e1b_lin)
print('\n')
print(m_e1a_pol, dm_e1a_pol)
print(m_e1b_pol, dm_e1b_pol)

