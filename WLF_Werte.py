# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:51:54 2022

@author: Simon
"""

import numpy as np
import pandas as pd

# Liste der geometrischen Maße aller Proben einer Messreihe
# Durchmesser d und Höhe h in m
d = np.array([0.00630, 0.00630, 0.00630, 0.00630, 0.00630,
              0.00630, 0.00630, 0.00630, 0.00630, 0.00630,
              0.00630, 0.00630, 0.00630, 0.00630, 0.00630,
              0.00630, 0.00630, 0.00630, 0.00630, 0.00630,])
h = np.array([0.00487, 0.00494, 0.00491, 0.00494, 0.00493,
              0.00496, 0.00481, 0.00491, 0.00491, 0.00498,
              0.00500, 0.00491, 0.00495, 0.00495, 0.00505,
              0.00495, 0.00491, 0.00499, 0.00492, 0.00499])

# Berechnung der nominalen Fläche für jede Probe
A = (np.pi/4)*(d**2)

# Liste der gemessenen Steigung des Schmelzpeaks aus der Messung mittels DSC
# Steigung der Probe sp und Steigung der Referenz sr in mW/°C
sp = np.array([-0.96, -0.91, -0.94, -0.95, -0.95,
               -0.93, -0.98, -0.92, -0.91, -0.94,
               -0.91, -0.91, -0.94, -0.93, -0.91,
               -0.94, -0.95, -0.91, -0.94, -0.91])
sr = np.array([-11.30, -11.30, -11.30, -11.30, -11.30,
               -11.30, -11.30, -11.30, -11.30, -11.30,
               -14.70, -14.70, -14.70, -14.70, -14.70,
               -14.70, -14.70, -14.70, -14.70, -14.70])

# Berechnung der Wärmeleitfähigkeit wlf, Mittelwert mean, Standardabweichung std
# und prozentualer Abstand prozent der Standardabweichung zum Mittelwert
wlf = h/(A*((1/(sr/1000))-(1/(sp/1000))))
mean = np.mean(wlf)
std = np.std(wlf)
prozent = (std/mean)*100

# Zusammenfassung aller Werte in einem Dataframe
data = pd.DataFrame({
    'd': d,
    'h': h,
    'A': A,
    'sp': sp,
    'sr': sr,
    'wlf': wlf})

# Abspeichern der Daten in einer csv-Datei
# data.to_csv(r'G:\Paper\Daten\PVC\data.csv', index=False)