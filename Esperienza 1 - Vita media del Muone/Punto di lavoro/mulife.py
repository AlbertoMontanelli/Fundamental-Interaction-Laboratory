import pylab
from matplotlib import pyplot as plt
import numpy as np
import os


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }


def dEff(n,N):
	deff = np.sqrt(n*(1-(n/N)))/N
	return deff
	
def dV(V):
	dV=V*(0.05/100)+1
	return dV

#pmt-02
V2,sin2,tr2,dop2= pylab.loadtxt('PMT-02.txt',unpack=True)
eff2 = tr2/dop2
rate2 = sin2/30

plt.figure("pmt02", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate2, eff2, dEff(tr2,dop2), np.sqrt(sin2)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V2,eff2,dEff(tr2,dop2), dV(V2) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT02", fontdict=font)
plt.savefig('pmt_02.png', dpi=300)



#pmt-03
V3,sin3,tr3,dop3= pylab.loadtxt('PMT-03a.txt',unpack=True)
eff3 = tr3/dop3
rate3 = sin3/30

plt.figure("pmt03", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate3, eff3, dEff(tr3,dop3), np.sqrt(sin3)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V3,eff3,dEff(tr3,dop3), dV(V3) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT03", fontdict=font)
plt.savefig('pmt_03.png', dpi=300)

#pmt-04
V4,sin4,tr4,dop4= pylab.loadtxt('PMT-04a.txt',unpack=True)
eff4 = tr4/dop4
rate4 = sin4/30

plt.figure("pmt04", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate4, eff4, dEff(tr4,dop4), np.sqrt(sin4)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V4,eff4,dEff(tr4,dop4), dV(V4) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT04", fontdict=font)
plt.savefig('pmt_04.png', dpi=300)

#pmt-05
V5,sin5,tr5,dop5= pylab.loadtxt('PMT-05a.txt',unpack=True)
eff5 = tr5/dop5
rate5 = sin5/30


plt.figure("pmt05", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate5, eff5, dEff(tr5,dop5), np.sqrt(sin5)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V5,eff5,dEff(tr5,dop5), dV(V5) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT05", fontdict=font)
plt.savefig('pmt_05.png', dpi=300)

#pmt-08
V8,sin8,tr8,dop8= pylab.loadtxt('PMT-08.txt',unpack=True)
eff8 = tr8/dop8
rate8 = sin8/30

plt.figure("pmt08", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate8, eff8, dEff(tr8,dop8), np.sqrt(sin8)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V8,eff8,dEff(tr8,dop8), dV(V8) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT08", fontdict=font)
plt.savefig('pmt_08.png', dpi=300)

#pmt-09
V9,sin9,tr9,dop9= pylab.loadtxt('PMT-09.txt',unpack=True)
eff9 = tr9/dop9
rate9 = sin9/30

plt.figure("pmt09", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate9, eff9, dEff(tr9,dop9), np.sqrt(sin9)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V9,eff9,dEff(tr9,dop9), dV(V9) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT09", fontdict=font)
plt.savefig('pmt_09.png', dpi=300)

#pmt-10

V10,sin10,tr10,dop10= pylab.loadtxt('PMT-10.txt',unpack=True)
eff10= tr10/dop10
rate10 = sin10/30


plt.figure("pmt10", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate10, eff10, dEff(tr10,dop10), np.sqrt(sin10)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V10,eff10,dEff(tr10,dop10), dV(V10) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT10", fontdict=font)
plt.savefig('pmt_10.png', dpi=300)

#pmt-11

V11,sin11,tr11,dop11= pylab.loadtxt('PMT-11.txt',unpack=True)
eff11 = tr11/dop11
rate11 = sin11/30

plt.figure("pmt11", dpi=300)
plt.subplot(1,2,1)
plt.errorbar(rate11, eff11, dEff(tr11,dop11), np.sqrt(sin11)/30,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.xlabel('Rate in singola [1/s]', fontdict=font)
plt.ylabel('Efficienza', fontdict=font)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.errorbar(V11,eff11,dEff(tr11,dop11), dV(V11) ,'o', color = 'blue', linestyle = ' ', ecolor='blue', elinewidth=0.8, markersize=2.5, capsize=2.5, capthick=0.8)
plt.grid()

plt.suptitle("Punto di lavoro per PMT11", fontdict=font)
plt.savefig('pmt_11.png', dpi=300)


