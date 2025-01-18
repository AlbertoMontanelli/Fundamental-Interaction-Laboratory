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
	
#pmt-01
V1,sin1,tr1,dop1= pylab.loadtxt('PMT-01.txt',unpack=True)
eff1 = tr1/dop1
rate1 = sin1/30
drate1=np.sqrt(sin1)/30
deff1=dEff(tr1, dop1)

#pmt-02
V2,sin2,tr2,dop2= pylab.loadtxt('PMT-02.txt',unpack=True)
eff2 = tr2/dop2
rate2 = sin2/30
drate2=np.sqrt(sin2)/30
deff2=dEff(tr2, dop2)

#pmt-03
V3, sin3, tr3, dop3 = pylab.loadtxt('PMT-03a.txt', unpack=True)
eff3 = tr3 / dop3
rate3 = sin3 / 30
drate3 = np.sqrt(sin3) / 30
deff3 = dEff(tr3, dop3)

#pmt-04
V4, sin4, tr4, dop4 = pylab.loadtxt('PMT-04a.txt', unpack=True)
eff4 = tr4 / dop4
rate4 = sin4 / 30
drate4 = np.sqrt(sin4) / 30
deff4 = dEff(tr4, dop4)

#pmt-05
V5, sin5, tr5, dop5 = pylab.loadtxt('PMT-05a.txt', unpack=True)
eff5 = tr5 / dop5
rate5 = sin5 / 30
drate5 = np.sqrt(sin5) / 30
deff5 = dEff(tr5, dop5)


#pmt-08
V8, sin8, tr8, dop8 = pylab.loadtxt('PMT-08.txt', unpack=True)
eff8 = tr8 / dop8
rate8 = sin8 / 30
drate8 = np.sqrt(sin8) / 30
deff8 = dEff(tr8, dop8)

#pmt-09
V9, sin9, tr9, dop9 = pylab.loadtxt('PMT-09.txt', unpack=True)
eff9 = tr9 / dop9
rate9 = sin9 / 30
drate9 = np.sqrt(sin9) / 30
deff9 = dEff(tr9, dop9)

#pmt-10
V10, sin10, tr10, dop10 = pylab.loadtxt('PMT-10.txt', unpack=True)
eff10 = tr10 / dop10
rate10 = sin10 / 30
drate10 = np.sqrt(sin10) / 30
deff10 = dEff(tr10, dop10)

#pmt-11
V11, sin11, tr11, dop11 = pylab.loadtxt('PMT-11.txt', unpack=True)
eff11 = tr11 / dop11
rate11 = sin11 / 30
drate11 = np.sqrt(sin11) / 30
deff11 = dEff(tr11, dop11)


print("PMT05")
for i in range(len(V5)):
	print("%.0f+-%.0f	%.0f+-%.0f	%.2f+-%.2f" %(V5[i], dV(V5)[i], rate5[i], drate5[i], eff5[i], deff5[i]))
	
	
print("\n")	
print("\n")	
	
	
print("PMT08")
for i in range(len(V8)):
	print("%.0f+-%.0f	%.0f+-%.0f	%.2f+-%.2f" %(V8[i], dV(V8)[i], rate8[i], drate8[i], eff8[i], deff8[i]))
	
	
print("\n")	
print("\n")	


print("PMT09")
for i in range(len(V9)):
	print("%.0f+-%.0f	%.0f+-%.0f	%.2f+-%.2f" %(V9[i], dV(V9)[i], rate9[i], drate9[i], eff9[i], deff9[i]))
	
	
print("\n")	
print("\n")


print("PMT10")
for i in range(len(V10)):
	print("%.0f+-%.0f	%.0f+-%.0f	%.2f+-%.2f" %(V10[i], dV(V10)[i], rate10[i], drate10[i], eff10[i], deff10[i]))
	
	
print("\n")	
print("\n")

print("PMT11")
for i in range(len(V11)):
	print("%.0f+-%.0f	%.0f+-%.0f	%.2f+-%.2f" %(V11[i], dV(V11)[i], rate11[i], drate11[i], eff11[i], deff11[i]))
	
	
print("\n")	
print("\n")
	
	
