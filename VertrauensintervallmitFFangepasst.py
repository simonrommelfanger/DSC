# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 17:08:51 2022

@author: Simon
"""

import itertools as iter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Einlesen der Datentabelle
data = pd.read_csv(r'E:\Paper\Daten\PVC\data.csv')

# Definition der benötigten Werte
n = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

d = np.array(data['d'])
h = np.array(data['h'])

sp = np.array(data['sp'])
sr = np.array(data['sr'])

p = np.array([0, 0, 0, 4.633, 5.281, 5.980, 6.169, 6.273, 6.245, 6.164,
              6.008, 5.755, 5.312, 4.673, 3.969, 3.242, 0, 0, 0, 0])
q = np.array([0, 0, 0, 3.359, 4.085, 4.791, 5.423, 5.860, 6.107, 6.259,
              6.335, 6.357, 6.244, 6.039, 5.329, 4.664, 0, 0, 0, 0])

# Festlegen des Messfehlers für die geometrischen Abmaße der Proben
deltam = 0.00002

# Werte des Faktors zur Berechnung des Vertrauensintervall
t99 = np.array([31.8205, 6.9646, 4.5407, 3.7469, 3.3649,
                3.1427, 2.9980, 2.8965, 2.8214, 2.7638,
                2.7181, 2.6810, 2.6503, 2.6245, 2.6025,
                2.5835, 2.5669, 2.5524, 2.5395, 2.5280])


mittelmax = np.array([])
mittelmin = np.array([])

# Berechnung WLF und Fehler
A = (np.pi/4)*(d**2)

wlf = abs(h/(A*((1/(sp/1000))-(1/(sr/1000)))))
mittelwert = np.mean(wlf)
std = np.std(wlf)

deltaA = (np.pi/2)*d*deltam
deltawlfh = (1/(A*((1/(sp/1000))-(1/(sr/1000)))))*deltam
deltawlfA = (h/(-(A**2)*((1/(sp/1000))-(1/(sr/1000)))))*deltaA
deltawlf = abs(deltawlfh) + abs(deltawlfA)

fehlereinzeln = std + deltawlf
fehler = np.mean(fehlereinzeln)
prozent = (fehler/mittelwert)*100
prozentstd = (std/mittelwert)*100
prozentdelta = (np.mean(deltawlf)/mittelwert)*100

print('WLF: '+str(mittelwert)+' +/-'+str(fehler)+' ('+str(prozent)+'%)')
print('STD: '+str(std)+' ('+str(prozentstd)+'%)', 'DELTAWLF: '+str(np.mean(deltawlf))+' ('+str(prozentdelta)+'%)')

# Berechnung Werte für Standard-Beta-Verteilung
muprozent = p/(p+q)
sigma2prozent = (p*q)/(((p+q)**2)*(p+q+1))
sigmaprozent = np.sqrt(sigma2prozent)

# Berechnung angepasste p-/q-Werte
treffer = 0
wlfoben = np.mean(wlf) + np.std(wlf)
wlfunten = np.mean(wlf) - np.std(wlf)
for i in range(len(wlf)):
    z = wlf[i]
    if z <= wlfoben:
        if z >= wlfunten:
            treffer = treffer + 1

print(treffer)
pangepasst = p + treffer
qangepasst = q + (len(wlf)-treffer)

# Berechnung Werte für Standard-Beta-Verteilung
muprozent = p/(p+q)
sigma2prozent = (p*q)/(((p+q)**2)*(p+q+1))
sigmaprozent = np.sqrt(sigma2prozent)

muprozentangepasst = pangepasst/(pangepasst+qangepasst)
sigma2prozentangepasst = (pangepasst*qangepasst)/(((pangepasst+qangepasst)**2)*(pangepasst+qangepasst+1))
sigmaprozentangepasst = np.sqrt((pangepasst*qangepasst)/(((pangepasst+qangepasst)**2)*(pangepasst+qangepasst+1)))

# Berechnung Mittelwerte für alle Probenanzahlen
for i in range(len(wlf)):
    mittelwerte = np.array([])
    x = iter.combinations(wlf, i+1)
    liste = list(x)
    for j in range(len(liste)):
        mittelwerte = np.append(mittelwerte, np.mean(liste[j]))
    mittelmax = np.append(mittelmax, max(mittelwerte))
    mittelmin = np.append(mittelmin, min(mittelwerte))
    print(i)

# Berechnung Erwartungswert μ und dazugehöriger Fehler σ
mu = ((mittelmax-mittelmin)*muprozentangepasst)+mittelmin
sigma = (mittelmax-mittelmin)*sigmaprozentangepasst
sigmaoben = mu + sigma
sigmaunten = mu - sigma
sigmaf = sigma + np.mean(deltawlf)
sigmafoben = mu + sigmaf
sigmafunten = mu - sigmaf

# Berechnung 99%iges Vertrauensinterval
vertrauen99 = t99*(sigmaf/np.sqrt(n))
vertrauen99oben = mu + vertrauen99
vertrauen99unten = mu - vertrauen99

# Berechnung für Tabelle
mup = ((mu-mittelwert)/mittelwert)*100
sigmafp = ((sigmaf/mu)*100)*2
vertrauen99p = ((vertrauen99/mu)*100)*2

# Erstellen des Grundgraphen mit Mittelwertmarkierung für N
plt.figure(figsize = [12, 8])
plt.grid('on')
plt.title('Erwartungswert mit Vertrauensintervallen PVC angepasst', weight = 'bold')
plt.xlabel('Anzahl der Proben', weight = 'bold')
plt.ylabel('Wärmeleitfähigkeit in W/(m*K)', weight = 'bold')
plt.axis(xmin=0, xmax=20, ymin=0.15, ymax=0.17)
plt.axhline(y=mittelwert, color='b', linewidth=2, linestyle='-')

# Einfügen der Ergebnisse der Berechnung in Graphen
plt.plot(n, mu, 'r',
         # n, sigmafoben, 'k',
         # n, sigmafunten, 'k',
         n, vertrauen99oben, 'c',
         n, vertrauen99unten, 'c')

# plt.fill_between(n, sigmafoben, sigmafunten, color='k', alpha = .1, label='Sigma + DeltaWLF')
plt.fill_between(n, vertrauen99oben, vertrauen99unten, color='c', alpha=.2, label='Konfidenz99%')
plt.legend(loc = 'upper right')
# plt.savefig(r'G:\Paper\Daten\PVC\05-Vertrauensintervall\VertrauenFFT99%angepasst.png', bbox_inches='tight', dpi=400)
plt.show()

np.savetxt(r'G:\Paper\Daten\PVC\05-Vertrauensintervall\wertefft99angepasst.txt', vertrauen99p)

# Zusammenfassung aller berechneten Werte in einem Dataframe
# tabelle = pd.DataFrame({
#     'Probenanzahl': n,
#     'Mittelwert': mittelwert,
#     'STD+Fehler': fehler,
#     'Mu': mu,
#     'Mu % Abstand': mup,
#     'Sigma+Fehler': sigmaf,
#     'Sigma % Abstand': sigmafp,
#     'Vertrauen90': vertrauen90,
#     'Vertrauen90 % Abstand': vertrauen90p,
#     'Vertrauen95': vertrauen95,
#     'Vertrauen95 % Abstand': vertrauen95p,
#     'Vertrauen97': vertrauen97,
#     'Vertrauen97 % Abstand': vertrauen97p,
#     'Vertrauen99': vertrauen99,
#     'Vertrauen99 % Abstand': vertrauen99p})
# tabelle.to_csv(r'E:\Laptop\Python\PMMA\VertrauenFPMMAangepasst.csv', index=False)