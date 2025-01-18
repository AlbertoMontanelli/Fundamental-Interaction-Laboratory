import matplotlib.pyplot as plt

# Dati delle misurazioni
misurazioni = ['28-29/02/2024', '5-6/03/2024', '07/03/2024', '12-13/03/2024', '19/03/2024', '19-20/03/2024', '20-21/03/2024','21/03/2024']
m1 = [517,410, 451,338,475,485, 548,379]
dm1 =[73,152,50,53,51,53,146,58]
m2 =[561,414,489,389,511, 526 , 596,391]
dm2 = [80,141,54,61,55,57,158,59]

# Valore atteso
valore_atteso = 511

# Modifica del font della label nella legenda
font = {'family': 'serif', 'weight': 'normal', 'size': 24}

# Creazione del plot
plt.figure('Picco 1')
plt.errorbar(m1,misurazioni,None,dm1,'o', color='blue', label='Massa misurata')
plt.axvline(x=511, color='red', linestyle='-', label='Massa elettrone', ymin=0.01, ymax=0.99)
plt.grid()
# Aggiunta delle etichette
plt.ylabel('Data acquisizione',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.xlabel('Massa dell\'elettrone [KeV]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.title('Confronto tra misure della massa dell\'elettrone e il valore atteso per il picco 1 del Co',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.legend(prop=font, labelcolor='black')
# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/Picco1.pdf', dpi=1200, bbox_inches='tight')


plt.figure('Picco2')
plt.errorbar( m2,misurazioni,None,dm2,'o', color='blue', label='Massa misurata')
plt.axvline(x=511, color='red', linestyle='-', label='Massa elettrone', ymin=0.01, ymax=0.99)
plt.grid()
# Aggiunta delle etichette
plt.ylabel('Data acquisizione',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.xlabel('Massa dell\'elettrone [KeV]',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.title('Confronto tra misure della massa dell\'elettrone e il valore atteso per il picco 2 del Co ',fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal','size':20,})
plt.legend(prop=font, labelcolor='black')
# Imposta la finestra del grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Pausa per permettere il rendering a schermo intero
plt.pause(2)  # Pausa di 2 secondi




plt.savefig('grafici/Picco2.pdf', dpi=1200, bbox_inches='tight')

# Mostra il grafico
plt.show()
