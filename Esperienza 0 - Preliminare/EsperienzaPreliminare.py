import numpy as np
import pylab as plb
import matplotlib.pyplot as plt

#font per grafici
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 13,
        }

##V TRIGGER VS FREQ. DI TRIGGER - PMT05

#data
Vtrig, f1, f2, f3 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/FreqdiTrigger.txt', unpack= True)

f=(f1+f2+f3)/3
df=np.sqrt(((f1-f)**2+(f2-f)**2+(f3-f)**2)/2)

print('\n')
for i in range (0,len(f)):
    print('for V=%f -> frequenza di trigger=%i+-%i' % (Vtrig[i],f[i], df[i]))

 
print('\n')   
#grafico
plt.figure('Frequenze di trigger', dpi=300)
plt.errorbar (Vtrig, f, df,  marker = 'o', color = 'darkred', linestyle = ' ', ecolor='red', elinewidth=1.5, markersize=2.5, capsize=4, capthick=1)
plt.ylabel('Frequenza di trigger [Hz]', fontdict=font)
plt.xlabel('Tensione di trigger [mV]', fontdict=font)
plt.grid(color = 'gray')
plt.savefig('freqditrigger.png', dpi=300)





##RITARDO SEGNALE ANALOGICO VS DISCRIMINATO
t1=15
t2=14.6
t3=16.2
t=(t1+t2+t3)/3
dt=np.sqrt(((t1-t)**2+(t2-t)**2+(t3-t)**2)/2)
#print(t, dt)



##CONTEGGI SINGOLI PMT05, PMT04, PMT03


#larghezza segnali discriminati
w05=np.array([44, 43.2, 45.6])
w05m=np.sum(w05)/3
dw05=np.sqrt(np.sum((w05-w05m)**2))
print('La larghezza del segnale discriminato di PMT05 è T= %0.2f +- %0.2f' % (w05m, dw05))
w04=np.array([46, 42, 46])
w04m=np.sum(w04)/3
dw04=np.sqrt(np.sum((w04-w04m)**2))
print('La larghezza del segnale discriminato di PMT04 è T= %0.2f +- %0.2f' % (w04m, dw04))
w03=np.array([50, 51, 56])
w03m=np.sum(w03)/3
dw03=np.sqrt(np.sum((w03-w03m)**2))
print('La larghezza del segnale discriminato di PMT03 è T= %0.2f +- %0.2f' % (w03m, dw03))
print('\n')

#misura di frequenza contatore vs oscilloscopio 
fosc05=np.array([91, 84, 64, 131, 92, 79])
fcont05=np.array([1260, 1231, 1234, 1236, 1202])
foscm05=np.sum(fosc05)/6
dfosc05=np.sqrt(np.sum((fosc05-foscm05)**2))
fcont05=np.sum(fcont05)/50
dfcont05=np.sqrt(fcont05)
fcont04=np.array([884, 820, 837, 829, 871])
fcont04=np.sum(fcont04)/50
dfcont04=np.sqrt(fcont04)
fcont03=np.array([980, 948, 973, 959, 974])
fcont03=np.sum(fcont03)/50
dfcont03=np.sqrt(fcont03)
print('La frequenza di trigger di PMT05 misurata con oscilloscopio è %i+-%i' % (foscm05, dfosc05))
print('La frequenza di trigger al contatore di PMT05 è %.0f+-%.0f' %(fcont05, dfcont05))
print('La frequenza di trigger al contatore di PMT04 è %.0f+-%.0f' %(fcont04, dfcont04))
print('La frequenza di trigger al contatore di PMT03 è %.0f+-%.0f' %(fcont03, dfcont03))
print('\n')

#data
V05, N05 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/ConteggiSingoliPMT05.txt', unpack= True)
V04, N04 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/ConteggiSingoliPMT04.txt', unpack= True)
V03, N03 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/ConteggiSingoliPMT03.txt', unpack= True)
dV05=V05*(0.05/100)+1
dV04=V04*(0.05/100)+1
dV03=V03*(0.05/100)+1
dN05=np.sqrt(N05)/10
dN04=np.sqrt(N04)/10
dN03=np.sqrt(N03)/10
N05=N05/10
N04=N04/10
N03=N03/10


print('PMT05')
for i in range (0,len(V05)):
    print('for V=%0.2f +-%0.2f -> #n=%0.2f+-%0.2f' % (V05[i],dV05[i],N05[i], dN05[i]))
    
print('\n')    

print('PMT04')
for i in range (0,len(V04)):
    print('for V=%0.2f +-%0.2f -> #n=%0.2f+-%0.2f' % (V04[i],dV04[i],N04[i], dN04[i]))
    
print('\n')
print('PMT03')
for i in range (0,len(V03)):
    print('for V=%0.2f +-%0.2f -> #n=%0.2f+-%0.2f' % (V03[i],dV03[i],N03[i], dN03[i]))

print('\n')

#grafico
plt.figure('Conteggi singoli PMT05', dpi=300)
plt.errorbar (V05, N05, dN05, dV05, marker = 'o', color = 'darkred', linestyle = ' ', ecolor='red', elinewidth=0.9, markersize=1.5, capsize=2, capthick=0.5)
plt.ylabel('Rate di conteggi [1/s]', fontdict=font)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.grid(color = 'gray', linestyle='-')
plt.savefig('ConteggisingoliPMT05.png', dpi=300)


plt.figure('Conteggi singoli PMT04', dpi=300)
plt.errorbar (V04, N04, dN04, dV04, marker = 'o', color = 'darkred', linestyle = ' ', ecolor='red',elinewidth=0.9, markersize=1.5, capsize=2, capthick=0.5)
plt.ylabel('Rate di conteggi [1/s]', fontdict=font)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.grid(color = 'gray')
plt.savefig('ConteggisingoliPMT04.png', dpi=300)


plt.figure('Conteggi singoli PMT03', dpi=300)
plt.errorbar (V03, N03, dN03, dV03, marker = 'o', color = 'darkred', linestyle = ' ', ecolor='red', elinewidth=0.9, markersize=1.5, capsize=2, capthick=0.5)
plt.ylabel('Rate di conteggi [1/s]', fontdict=font)
plt.xlabel('Tensione di alimentazione [V]', fontdict=font)
plt.grid(color = 'gray')
plt.savefig('ConteggisingoliPMT03.png', dpi=300)


##EFFICIENZE

Doppie03, Triple03, Singola05di03, Singola04di03, Singola03di03 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/EfficienzaPMT03.txt', unpack= True)
Doppie04, Triple04, Singola05di04, Singola04di04, Singola03di04 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/EfficienzaPMT04.txt', unpack= True)
Doppie05, Triple05, Singola05di04, Singola04di04, Singola03di04 =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/EfficienzaPMT05.txt', unpack= True)


Triple03sum=np.sum(Triple03)
Doppie03sum=np.sum(Doppie03)
Eff03sum=Triple03sum/Doppie03sum
dEff03sum=np.sqrt( (Eff03sum*(1-Eff03sum))/Doppie03sum)

Eff03=Triple03/Doppie03
Eff03m=(np.sum(Eff03))/10
dEff03m=np.sqrt(np.sum((Eff03-Eff03m)**2))

print('L\'efficienza di PMT03 con statistica binomiale è %f+-%f' %(Eff03sum, dEff03sum))
print('L\'efficienza di PMT03 con media e dev. standard è %f+-%f' %(Eff03m, dEff03m))
print('\n')

Triple04sum=np.sum(Triple04)
Doppie04sum=np.sum(Doppie04)
Eff04sum=Triple04sum/Doppie04sum
dEff04sum=np.sqrt( (Eff04sum*(1-Eff04sum))/Doppie04sum)

Eff04=Triple04/Doppie04
Eff04m=(np.sum(Eff04))/10
dEff04m=np.sqrt(np.sum((Eff04-Eff04m)**2))

print('L\'efficienza di PMT04 con statistica binomiale è %f+-%f' %(Eff04sum, dEff04sum))
print('L\'efficienza di PMT04 con media e dev. standard è %f+-%f' %(Eff04m, dEff04m))
print('\n')

Triple05sum=np.sum(Triple05)
Doppie05sum=np.sum(Doppie05)
Eff05sum=Triple05sum/Doppie05sum
dEff05sum=np.sqrt( (Eff05sum*(1-Eff05sum))/Doppie05sum)

Eff05=Triple05/Doppie05
Eff05m=(np.sum(Eff05))/10
dEff05m=np.sqrt(np.sum((Eff05-Eff05m)**2))

print('L\'efficienza di PMT05 con statistica binomiale è %f+-%f' %(Eff05sum, dEff05sum))
print('L\'efficienza di PMT05 con media e dev. standard è %f+-%f' %(Eff05m, dEff05m))
print('\n')



##EFFICIENZA PMT04 AL VARIARE DELLA TENSIONE DI ALIMENTAZIONE


HTV, Doppie04HTV, Triple04HTV, Singola05HTV, Singola04HTV, Singola03HTV =plb.loadtxt('/home/alberto/Documenti/Laboratorio Interazioni Fondamentali/Esperienza Preliminare/Dati/EfficienzaPMT04vsHTV.txt', unpack= True)
Eff04HTV=Triple04HTV/Doppie04HTV
dEff04HTV=np.sqrt( (Eff04HTV*(1-Eff04HTV))/Doppie04HTV)
dHTV=HTV*(0.05/100)+1

for i in range (0,len(HTV)):
    print('for V=%.2f +- %0.2f -> Efficienza =%.2f+-%.2f' % (HTV[i], dHTV[i], Eff04HTV[i]*100, dEff04HTV[i]*100))
    
print('\n')    
    
#grafico
plt.figure('Efficienza PMT04 vs Tensione di alimentazione', dpi=300)
plt.errorbar (HTV, Eff04HTV*100, dEff04HTV*100, dHTV,  marker = 'o', ecolor='red', color = 'darkred', linestyle = ' ', elinewidth=1, markersize=1.1)
plt.ylabel('Efficienza PMT04', fontdict=font)
plt.xlabel('Tensione di alimentazione PMT04 [V]', fontdict=font)
plt.grid(color = 'gray')
plt.savefig('EffPMT04_vs_HTV.png', dpi=300)
plt.show()

