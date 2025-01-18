import pylab as plb
from matplotlib import pyplot as plt
import numpy as np
import os
from ROOT import *


dt1= plb.loadtxt('FakeStart.txt',unpack=True)
dt2= plb.loadtxt('FakeStop.txt',unpack=True)

	
c=TCanvas('Fake Start and Stop')

c.Divide(2)
c.cd(1)

i=0			
hist1=TH1F("hist", "Fake Start", 150, 0, 20e-6)
for i in range (len(dt1)):
	hist1.Fill(dt1[i])
hist1.GetXaxis().SetTitle("Tempo [s]")
hist1.GetYaxis().SetTitle("Eventi")
hist1.Draw()









c.cd(2)
hist2=TH1F("hist", "Fake Stop", 150, 0, 20e-6)
i=0
for i in range (len(dt2)):
	hist2.Fill(dt2[i])
hist2.GetXaxis().SetTitle("Tempo [s]")
#hist2.GetYaxis().SetTitle("Eventi")
hist2.Draw()

#c.SaveAs("FakeStartStop.png")

input()
