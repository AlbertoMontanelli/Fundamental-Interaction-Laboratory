import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
import textwrap
from scipy.optimize import curve_fit
from ROOT import *

def convertitore(nome_file):

	# Apre il file in modalità di lettura
	with open(nome_file, "r") as file:
    		# Legge le righe del file
    		dati_esadecimali = file.readlines()
	
	# Rimuove eventuali spazi bianchi o caratteri di nuova riga
	dati_esadecimali = [line.strip() for line in dati_esadecimali]
	# Converte i dati esadecimali in numeri decimali
	dati_decimali = [int(num, 16) for num in dati_esadecimali]
	
	return dati_decimali


# per richiamare -> from nomefile (senza .py) import nomefunzione

