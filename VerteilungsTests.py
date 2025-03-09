import itertools as iter
import numpy as np
import pandas as pd
from scipy import stats as scipy

# Einlesen der Daten
data = pd.read_csv(r'E:\Paper\Daten\2.Iteration\data.csv')
werte = data['wlf']

# Definition der notwendigen Variablen
n = np.array([])
statistic_ks = np.array([])
pvalue_ks = np.array([])
statistic_sw = np.array([])
pvalue_sw = np.array([])
pvalue_ad = np.array([])
grenze_ad = np.array([])

# Berechnung aller möglichen Kombinationen in Abhängigkeit der Probenzahl
# verschiedene Test auf Normalverteilung
for i in range (len(werte)-1):
    mittelwerte = np.array([])
    n = np. append(n, i+1)
    x = iter.combinations(werte, i+1)
    liste = list(x)
    for j in range(len(liste)):
        mittelwerte = np.append(mittelwerte, np.mean(liste[j]))
    stat_sw, p_sw = scipy.shapiro(mittelwerte)
    statistic_sw = np.append(statistic_sw, stat_sw)
    pvalue_sw = np.append(pvalue_sw, p_sw)
    stat_ks, p_ks = scipy.kstest(mittelwerte, 'norm')
    statistic_ks = np.append(statistic_ks, stat_ks)
    pvalue_ks = np.append(pvalue_ks, p_ks)
    stat_ad = scipy.anderson(mittelwerte, 'norm')
    pvalue_ad = np.append(pvalue_ad, stat_ad[0])
    grenze_ad = np.append(grenze_ad, stat_ad[1][2])
    print(max(n))

# Zusammenfassen aller berechneten Daten
tabelle = pd.DataFrame({
    'Probenanzahl': n,
    'StatisticResult KS': statistic_ks,
    'P-Value KS': pvalue_ks,
    'StatisticResult SW': statistic_sw,
    'P-Value SW': pvalue_sw,
    'Grenzwert AD': grenze_ad,
    'P-Value AD': pvalue_ad})