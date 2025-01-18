import pylab
from matplotlib import pyplot as plt
import numpy as np
import os

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }





#CALIBRAZIONE FPGA
chc, tc= pylab.loadtxt('MV_calibrazione1.dat',unpack=True)

tem = np.array([])

for i in range(len(tc)):
	a = tc[i]-tc[i-1]
	tem = np.append(tem,a)

print(tem)
freq = 1/tem
plt.figure('2')
plt.hist(freq, bins = 1000, range=(286.3,286.4),edgecolor='blue')

plt.xlabel('freq')
plt.ylabel('eventi')
plt.title('Istogramma1')


plt.show()






