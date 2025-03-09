# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:59:21 2022

@author: Simon
"""

# import math
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import pandas as pd
from scipy import interpolate
from scipy.integrate import quad
import scipy.signal
# from scipy import stats as stats
import tkinter as tk
import tkinter
from tkinter import filedialog




#%%----------------------------------------------------------------------------


def select_file():
  """
  ################
  ### Complete ###
  ################

  Returns
  -------
  file_root : TYPE
    DESCRIPTION.

  """
  root = tk.Tk()
  root.attributes('-topmost',True)
  root.wm_attributes('-topmost', 1)
  root.withdraw()
  file_root = filedialog.askopenfilename().split('/')
  file_root = os.path.join(file_root[0],os.sep,*file_root[1:])

  return file_root


#%%----------------------------------------------------------------------------

def read_txt_höhenprofil(file, plot, detilt):
    head, tail = ntpath.split(file)
    name = tail.strip('.txt')
    
    b = np.array([])
    
    # open and read file as lines of string
    with open(file, 'r') as f:
        content = f.readlines()[15::]

    data_np = np.zeros((len(content), 2), dtype=object)

    # separate different entries in line and put them in data as string
    for i in range(len(content)):
        a = content[i]
        b = np.append(b, np.array(np.char.split(a)))
        data_np[i] = b[i]
    
    # convert every entry from string to float 
    for j in range(np.shape(data_np)[0]):
        for k in range(np.shape(data_np)[1]):
            data_np[j][k] = float(data_np[j][k].replace(',', '.'))


    # convert array in dataframe
    x = data_np[:,0]
    z = data_np[:,1]
    z_test = list(data_np[:,1])
    z_test_nan = np.isnan(z_test)
    
    index = list()
    for i in range(len(z_test)):
        if z_test[i] == 0:
            index.append(int(i))
        elif z_test_nan[i] == True:
            index.append(int(i))
    
    x = np.delete(x, index)
    z = np.delete(z, index)
    
    peaks, _ = scipy.signal.find_peaks(z)
    
    
    if detilt == True:
        xmax = np.max(x)
        mean_vorne = np.mean(z[:61])
        mean_hinten = np.mean(z[-61:])
        tilt = mean_vorne - mean_hinten
        z_detilted = z+(tilt*(x/xmax))
        
        peaks_detilted, _ = scipy.signal.find_peaks(z_detilted)
    
    else:
        pass
    
    
    
    if plot == True:
        plt.figure(figsize=[13,10])
        plt.plot(x, z, label = 'Höhenprofil')
        plt.plot(x[peaks], z[peaks], 'x', label = 'Peaks')
        plt.title('Höhenprofil '+str(name), size=25, weight='bold')
        plt.xlabel('Höhe [m]', size=25)
        plt.ylabel('Messstrecke [m]', size=25)
        
        plt.grid('on', linestyle='--')
        plt.tick_params(labelsize = 25)
        legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
        plt.legend(prop=legend_font, loc = 'upper right')
        plt.show()
        
        if detilt == True:
            plt.figure(figsize=[13,10])
            plt.plot(x, z_detilted, label = 'Höhenprofil detilted')
            plt.plot(x[peaks_detilted], z_detilted[peaks_detilted], 'x', label = 'Peaks')
            plt.title('Höhenprofil '+str(name)+' detilted', size=25, weight='bold')
            plt.xlabel('Höhe [m]', size=25)
            plt.ylabel('Messstrecke [m]', size=25)
            
            plt.grid('on', linestyle='--')
            plt.tick_params(labelsize = 25)
            legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
            plt.legend(prop=legend_font, loc = 'upper right')
            plt.show()
            
        else:
            pass
        
    else:
        pass
    
    # convert array in dataframe
    if detilt == True:
        d = {'x': x, 'z': z, 'z_detilted': z_detilted}
        data = pd.DataFrame(data=d)
        
        return data, peaks, peaks_detilted, name
        
    else:
        d = {'x': x, 'z': z}
        data = pd.DataFrame(data=d)
        
        return data, peaks, name
    


#%%----------------------------------------------------------------------------

def get_hist(data, name):
    if len(data) <=1000:
        a = np.sqrt(len(data))
    else:
        a = 10*np.log(len(data))
    b = int(a)
    
    plt.figure(figsize=[13,10])
    count, bins_object, ignored = plt.hist(data, bins = b, density=True, visible=False)
    
    bins = np.array([])
    for i in range(len(bins_object)):
        bins = np.append(bins, bins_object[i])
    
    mu = np.mean(data)
    sigma = np.std(data)
    phi = scipy.stats.norm.pdf(bins, mu, sigma)
    
    plt.plot(bins, phi, color='r')
    plt.title('Histogramm für Höhenprofil '+str(name)+' detilted', size=25, weight='bold')
    plt.xlabel('Höhe des Profil[m]', size=25)
    plt.ylabel('Anzahl der Peaks', size=25)
    
    plt.grid('on', linestyle='--')
    plt.tick_params(labelsize = 25)
    # legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
    # plt.legend(prop=legend_font, loc = 'upper right')
    plt.show()
    
    return b, count, bins, phi


#%%----------------------------------------------------------------------------

def get_distribution(bins):
    
    phi = scipy.stats.norm.pdf(bins, mu, sigma)
    return phi



#%%----------------------------------------------------------------------------


def get_peak_radius(x_data, z_data, peaks):
    peak_radius = np.array([])
    
    for i in range(len(peaks)):
        x_minus = x_data[peaks[i]-1]
        x_peak = x_data[peaks[i]]
        x_plus = x_data[peaks[i]+1]
        
        z_minus = z_data[peaks[i]-1]
        z_peak = z_data[peaks[i]]
        z_plus = z_data[peaks[i]+1]
        
        c_kreis = (x_peak-x_minus)**2 + (z_peak-z_minus)**2
        a_kreis = (x_minus-x_plus)**2 + (z_minus-z_plus)**2
        b_kreis = (x_plus-x_peak)**2 + (z_plus-z_peak)**2
        ar = a_kreis**0.5
        br = b_kreis**0.5
        cr = c_kreis**0.5 
        r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
        peak_radius = np.append(peak_radius, r)
        
    return peak_radius


#%%----------------------------------------------------------------------------

def get_real_area(peak_density, area_nominal, peak_radius, sigma, f):
    
    real_area = np.pi*peak_density*area_nominal*peak_radius*sigma*f
    
    return real_area

#%%----------------------------------------------------------------------------

def get_wlf(h, A, sp, sr):
    
    wlf_einzeln = abs(h/(A*((1/(sp/1000))-(1/(sr/1000)))))
    
    wlf_mean = np.mean(wlf_einzeln)
    std_mean = np.std(wlf_einzeln)
    
    return wlf_einzeln, wlf_mean, std_mean

#%%----------------------------------------------------------------------------

def get_3dplot(x_data, y_data, z_data1, z_data2, title):
    
    x_20 = 2
    y_20 = 0.30
    # text_20 = ('20er Messreihe')
    # text_max = ('Max. Abstand')
    
    x_lim = (np.round(min(x_data),3), np.round(max(x_data),3))
    y_lim = (np.round(min(y_data),3), np.round(max(y_data),3))
    
    z1_min = min(z_data1)
    z1_max = max(z_data1)
    z2_min = min(z_data2)
    z2_max = max(z_data2)
    
    if z1_min < z2_min:
        z_min = z1_min
    else:
        z_min = z2_min
    if z1_max > z2_max:
        z_max = z1_max
    else:
        z_max = z2_max
    
    z_lim = (np.round(z_min,3), np.round(z_max,3))
    
    x_2d, y_2d = np.meshgrid(x_data, y_data)
    
    # y_2d = np.array([[0.77, 0.91, 0.88],
    #                  [0.42, 0.37, 0.35],
    #                  [0.14, 0.21, 0.09]])
    
    z_values1 = np.array([[z_data1[0], z_data1[3], z_data1[6]],
                          [z_data1[1], z_data1[4], z_data1[7]],
                          [z_data1[2], z_data1[5], z_data1[8]]])

    z_values2 = np.array([[z_data2[0], z_data2[3], z_data2[6]],
                          [z_data2[1], z_data2[4], z_data2[7]],
                          [z_data2[2], z_data2[5], z_data2[8]]])    


    z_value_bottom = np.array([[z_lim[0], z_lim[0], z_lim[0]],
                               [z_lim[0], z_lim[0], z_lim[0]],
                               [z_lim[0], z_lim[0], z_lim[0]]])
    
    z_values1_round = np.round(z_values1, 4)
    z_delta = z_values2-z_values1
    z_delta_round = np.round(z_delta, 4)
    
    
    values1_wlf = np.array([[str(z_values1_round[0][0]), str(z_values1_round[0][1]), str(z_values1_round[0][2])],
                            [str(z_values1_round[1][0]), str(z_values1_round[1][1]), str(z_values1_round[1][2])],
                            [str(z_values1_round[2][0]), str(z_values1_round[2][1]), str(z_values1_round[2][2])]])
    
    values_delta_wlf = np.array([['+'+str(z_delta_round[0][0]), '+'+str(z_delta_round[0][1]), '+'+str(z_delta_round[0][2])],
                                 ['+'+str(z_delta_round[1][0]), '+'+str(z_delta_round[1][1]), '+'+str(z_delta_round[1][2])],
                                 ['+'+str(z_delta_round[2][0]), '+'+str(z_delta_round[2][1]), '+'+str(z_delta_round[2][2])]])
    
    
    xnew, ynew = np.mgrid[1:5:100j, 0.1:0.79:100j]
    tck1 = interpolate.bisplrep(x_2d, y_2d, z_values1, s=0, kx=2, ky=2)
    z_values1_new = interpolate.bisplev(xnew[:,0], ynew[0,:], tck1)

    tck2 = interpolate.bisplrep(x_2d, y_2d, z_values2, s=0, kx=2, ky=2)
    z_values2_new = interpolate.bisplev(xnew[:,0], ynew[0,:], tck2)
    
    z_delta_new = z_values2_new - z_values1_new
    ind_delta_max = np.unravel_index(np.argmax(z_delta_new, axis=None), z_delta_new.shape)
    
    fig = plt.figure(figsize=[13,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x_2d, y_2d, z_values1, marker='x', color='black', linewidth=2)
    ax.plot_surface(xnew, ynew, z_values1_new, alpha=.6, color='gray')
    ax.scatter3D(x_2d, y_2d, z_values2, marker='x', color='black', linewidth=2)
    ax.plot_surface(xnew, ynew, z_values2_new, alpha=.6, cmap='jet')
    ax.scatter3D(x_2d, y_2d, z_value_bottom, marker='x', color='black', linewidth=2)
    ax.plot_wireframe(x_2d, y_2d, z_value_bottom, color='black', linestyle='--', alpha=.5)
    ax.plot_wireframe(x_2d, y_2d, z_values1, linestyle='--', color='k', alpha=.4)
    
    ax.scatter3D(x_20, y_20, z_lim[0], marker='.', color='green', linewidth=5)
    # ax.text(x_20, y_20, z_lim[0], text_20, weight='bold', va='top')
    ax.scatter3D(x_20, y_20, z_values2_new[25,29], marker='.', color='green', linewidth=5)
    
    ax.scatter3D((xnew[ind_delta_max[0]][ind_delta_max[1]], xnew[ind_delta_max[0]][ind_delta_max[1]]),\
                 (ynew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]]),\
                 (z_lim[0], z_lim[0]),\
                     marker='.', color='red', linewidth=5)
    ax.scatter3D((xnew[ind_delta_max[0]][ind_delta_max[1]], xnew[ind_delta_max[0]][ind_delta_max[1]]),\
                 (ynew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]]),\
                 (z_values2_new[ind_delta_max[0]][ind_delta_max[1]], z_values2_new[ind_delta_max[0]][ind_delta_max[1]]),\
                 marker='.', color='red', linewidth=5)
    # ax.text(xnew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]],\
    #    z_lim[0], text_max, weight='bold', va='top')
    
    # for i in range(3):
    #     for j in range(3):
    #         ax.text(x_2d[i][j], y_2d[i][j], z_values1[i][j], values1_wlf[i][j], weight='bold')
    #         ax.text(x_2d[i][j], y_2d[i][j], z_values2[i][j], values_delta_wlf[i][j], weight='bold')

    
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_zlim(z_lim[0], z_lim[1])
    # plt.title(title, size=25, weight='bold')
    ax.set_xlabel('specimen height $h$ [mm]', size=25, labelpad=20)
    ax.set_ylabel('surface roughness $R_a$ [μm]', size=25, labelpad=25)
    ax.set_zlabel('   thermal conductivity'
                  '\n'
                  '$λ$ [W/mK]', size=25, labelpad=45)
    plt.grid('on', linestyle='--')
    plt.tick_params(labelsize = 25)
    ax.tick_params(axis='z', which='major', pad=15)
    # plt.savefig(r'F:\Paper\Graphen\3Dplotshort.png', bbox_inches='tight', dpi=200)
    plt.show()

#%%----------------------------------------------------------------------------

def get_3dplot_prozent(x_data, y_data, z_data1, z_data2, ziel, faktor, title):
    
    x_20 = 2
    y_20 = 0.30
    # text_20 = ('20er Messreihe')
    # text_max = ('Max. Abstand')
    
    x_2d, y_2d = np.meshgrid(x_data, y_data)
    
    # y_2d = np.array([[0.77, 0.91, 0.88],
    #                  [0.42, 0.37, 0.35],
    #                  [0.14, 0.21, 0.09]])
    
    # y_2d_max = np.array([[0.91, 0.91, 0.91],
    #                      [0.37, 0.37, 0.37],
    #                      [0.09, 0.09, 0.09]])
    
    z_data_prozent = ((z_data2-z_data1)/z_data1)*100
    
    values_prozent = np.array([[z_data_prozent[0], z_data_prozent[3], z_data_prozent[6]],
                               [z_data_prozent[1], z_data_prozent[4], z_data_prozent[7]],
                               [z_data_prozent[2], z_data_prozent[5], z_data_prozent[8]]])
    
    values_prozent_round = np.round(values_prozent, 2)
    
    z_ziel = np.array([[ziel, ziel, ziel],
                       [ziel, ziel, ziel],
                       [ziel, ziel, ziel]])
    
    z_mit_faktor = z_ziel*faktor
    
    x_lim = (min(x_data), max(x_data))
    y_lim = (min(y_data), max(y_data))
    
    z_lim = (0, (ziel*faktor)+1)
    
    z_value_bottom = np.array([[z_lim[0], z_lim[0], z_lim[0]],
                               [z_lim[0], z_lim[0], z_lim[0]],
                               [z_lim[0], z_lim[0], z_lim[0]]])
    
    xnew, ynew = np.mgrid[1:5:100j, 0.1:0.79:100j]
    tck1 = interpolate.bisplrep(x_2d, y_2d, values_prozent, s=0, kx=2, ky=2)
    z_values_new = interpolate.bisplev(xnew[:,0], ynew[0,:], tck1)
    
    ind_delta_max = np.unravel_index(np.argmax(z_values_new, axis=None), z_values_new.shape)
    
    fig = plt.figure(figsize=[13,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x_2d, y_2d, z_value_bottom, linestyle='--', color='black', alpha=.5)
    ax.scatter3D(x_2d, y_2d, values_prozent, marker='x', color='black', linewidth=2)
    ax.plot_wireframe(x_2d, y_2d, values_prozent, linestyle='--', color='k', alpha=.5)
    ax.plot_surface(xnew, ynew, z_values_new, alpha=.6, cmap='jet')
    ax.plot_surface(x_2d, y_2d, z_ziel, alpha=0.2, color='black')
    ax.plot_surface(x_2d, y_2d, z_mit_faktor, alpha=0.2, color='green')
    ax.scatter3D(x_2d, y_2d, 0, marker='x', color='black', linewidth=2)
    
    ax.scatter3D(x_20, y_20, z_lim[0], marker='.', color='green', linewidth=5)
    # ax.text(x_20, y_20, z_lim[0], text_20, weight='bold', va='top')
    ax.scatter3D(x_20, y_20, z_values_new[25,29], marker='.', color='green', linewidth=5)
    
    # ax.scatter3D((xnew[ind_delta_max[0]][ind_delta_max[1]], xnew[ind_delta_max[0]][ind_delta_max[1]]),\
    #              (ynew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]]),\
    #              (z_lim[0], z_lim[0]),\
    #                  marker='.', color='red', linewidth=5)
    ax.scatter3D((1, 1),\
                 (0.44, 0.44),\
                     (z_lim[0], z_lim[0]),\
                         marker='.', color='red', linewidth=5)
    # ax.scatter3D((xnew[ind_delta_max[0]][ind_delta_max[1]], xnew[ind_delta_max[0]][ind_delta_max[1]]),\
    #              (ynew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]]),\
    #              (z_values_new[ind_delta_max[0]][ind_delta_max[1]], z_values_new[ind_delta_max[0]][ind_delta_max[1]]),\
    #                  marker='.', color='red', linewidth=5)
    ax.scatter3D((1, 1),\
                 (0.44, 0.44),\
                     (z_values_new[ind_delta_max[0]][ind_delta_max[1]], z_values_new[ind_delta_max[0]][ind_delta_max[1]]),\
                         marker='.', color='red', linewidth=5)
    # ax.text(xnew[ind_delta_max[0]][ind_delta_max[1]], ynew[ind_delta_max[0]][ind_delta_max[1]],\
    #    z_lim[0], text_max, weight='bold', va='top')
        
    for i in range(3):
        for j in range(3):
            ax.text(x_2d[i][j], y_2d[i][j], values_prozent[i][j], values_prozent_round[i][j], size=20, verticalalignment='center')
                
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_zlim(z_lim[0], z_lim[1])
    # plt.title(title, size=25, weight='bold')
    ax.set_xlabel('specimen height $h$ [mm]', size=25, labelpad=20)
    ax.set_ylabel('surface roughness $R_a$ [μm]', size=25, labelpad=20)
    ax.set_zlabel('   percantage distance'
                  '\n'
                  '$σ$ [%]', size=25, labelpad=20)
    plt.grid('on', linestyle='--')
    plt.tick_params(labelsize = 25)
    ax.tick_params(axis='z', which='major', pad=5)
    # plt.savefig(r'F:\Paper\Graphen\3Dplotprozent.png', bbox_inches='tight', dpi=200)
    plt.show()
    
#%%----------------------------------------------------------------------------
intervallgrenze = 30
messfläche = 5e-3 * 2e-6
höhe = ('1', '3', '5')
schleifpapier = ('600', '1200', '4000')
l = [1, 2, 3, 4, 5]
tabellen = ('1-600', '1-1200', '1-4000',
            '3-600', '3-1200', '3-4000',
            '5-600', '5-1200', '5-4000')
deltam = 0.00002

probenhöhe = np.array([])
rauheit = np.array([])
real_area = np.array([])
real_area_std = np.array([])
durchmesser = np.array([])
nominal_area = np.array([])
probensteigung = np.array([])
referenzsteigung = np.array([])
# delta_wlf_nominal = np.array([])
delta_wlf_real = np.array([])
delta_wlf_real_std = np.array([])
delta_area_nomimal = np.array([])
height = np.array([])

delta_wlf_nominal = np.array([])

wlf_einzeln_nominal = np.array([])
wlf_mean_nominal = np.array([])
std_mean_nominal = np.array([])

wlf_einzeln_real = np.array([])
wlf_mean_real = np.array([])
std_mean_real = np.array([])

real_area_one_check = np.array([])


for i in range(len(höhe)):
# for i in range(1):
    for j in range(len(schleifpapier)):
    # for j in range(1):
        delta_peak_radius = np.array([])
        peak_density = np.array([])
        peak_radius = np.array([])
        
        area_real = np.array([])
        std_area_real_10 = np.array([])
        # real_area_rauheit = np.array([])
        wert_rauheit = np.array([])
        
        d = np.loadtxt(r'G:\Paper\Daten\Fehlerfeld\Werte\d_'+str(höhe[i])+'-'+str(schleifpapier[j])+'.txt')
        h = np.loadtxt(r'G:\Paper\Daten\Fehlerfeld\Werte\h_'+str(höhe[i])+'-'+str(schleifpapier[j])+'.txt')
        sp = np.loadtxt(r'G:\Paper\Daten\Fehlerfeld\Werte\sp_'+str(höhe[i])+'-'+str(schleifpapier[j])+'.txt')
        sr = np.loadtxt(r'G:\Paper\Daten\Fehlerfeld\Werte\sr_'+str(höhe[i])+'-'+str(schleifpapier[j])+'.txt')
        
        a_nominal = (np.pi/4)*(d**2)
        
        delta_area_nomimal = (np.pi/2)*d*deltam
        delta_wlf_h_nominal = (1/(a_nominal*((1/(sp/1000))-(1/(sr/1000)))))*deltam
        delta_wlf_area_nominal = (h/(-(a_nominal**2)*((1/(sp/1000))-(1/(sr/1000)))))*delta_area_nomimal
        delta_wlf_nominal_einzeln = abs(delta_wlf_h_nominal) + abs(delta_wlf_area_nominal)
        delta_wlf_nominal = np.append(delta_wlf_nominal, np.mean(delta_wlf_nominal_einzeln))
        
        wlf_einzeln_nominal_z, wlf_mean_nominal_z, std_mean_nominal_z = get_wlf(h, a_nominal, sp, sr)
        
        wlf_einzeln_nominal = np.append(wlf_einzeln_nominal, wlf_einzeln_nominal_z)
        wlf_mean_nominal = np.append(wlf_mean_nominal, wlf_mean_nominal_z)
        std_mean_nominal = np.append(std_mean_nominal, std_mean_nominal_z)
        
        
        for k in range(len(d)):
        # for k in range(1):
            peak_density_z2 = np.array([])
            peak_radius_z = np.array([])
            
            area_nominal = a_nominal[k]
            
            real_area_one = np.array([])
            peak_number = np.array([])
            for m in range(len(l)):
            # for m in range(1):
                file = r'G:\HöhenProfil\P'+str(schleifpapier[j])+'_L-'+str(l[m])+'.txt'
            
                
                data, peaks, peaks_detilted, name = read_txt_höhenprofil(file, plot=False, detilt=True)
                
                x = data['x']
                z_detilted = data['z_detilted']
                
                b, count, bins, phi = get_hist(z_detilted, name)
                
                mu = np.mean(z_detilted)
                sigma = np.std(z_detilted)
                
                grenze = np.percentile(z_detilted, intervallgrenze)
                
                res, err = quad(get_distribution, grenze, max(bins))
                
                peak_radii = get_peak_radius(x, z_detilted, peaks_detilted)
                peak_radius_z = np.append(peak_radius_z, np.mean(peak_radii))

                
                peak_number = np.append(peak_number, len(peaks_detilted))
                peak_density_z = len(peaks_detilted)/messfläche
                peak_density_z2 = np.append(peak_density_z2, peak_density_z)
                
                
                value_real_area = get_real_area(peak_density_z, a_nominal[k], peak_radius_z, sigma, res)
                real_area_one = np.append(real_area_one, value_real_area)
                # real_area_one_check = np.append(real_area_one_check, get_real_area(peak_density_z, a_nominal[k], peak_radius, sigma, res))
                
                print(i, j, k, m)
            
            
            peak_radius = np.append(peak_radius, np.mean(peak_radius_z))
            delta_peak_radius = np.append(delta_peak_radius, np.std(peak_radii))
            
            peak_density = np.append(peak_density, np.mean(peak_density_z2))
            
            area_real = np.append(area_real, np.mean(real_area_one)) 
            std_area_real_10 = np.append(std_area_real_10, np.std(real_area_one))
            # real_area_rauheit = np.append(real_area_rauheit, np.mean(real_area_one))
            wert_rauheit = np.append(wert_rauheit, schleifpapier[j])
            delta_peak_density = np.std(peak_number) * messfläche

        
        wlf_einzeln_real_z, wlf_mean_real_z, std_mean_real_z = get_wlf(h, area_real, sp, sr)
        
        wlf_einzeln_real = np.append(wlf_einzeln_real, wlf_einzeln_real_z)
        wlf_mean_real = np.append(wlf_mean_real, wlf_mean_real_z)
        std_mean_real = np.append(std_mean_real, std_mean_real_z)

        rauheit = np.append(rauheit, wert_rauheit)
        real_area = np.append(real_area, area_real)
        real_area_std = np.append(real_area_std, std_area_real_10)
        durchmesser = np.append(durchmesser, d)
        probenhöhe = np.append(probenhöhe, h)
        nominal_area = np.append(nominal_area, a_nominal)
        probensteigung = np.append(probensteigung, sp)
        referenzsteigung = np.append(referenzsteigung, sr)
        height = np.append(height, h)
        
        delta_area_real = np.pi*peak_density*a_nominal*sigma*res*delta_peak_radius + \
            np.pi*peak_density*peak_radius*sigma*res*delta_area_nomimal + \
            np.pi*a_nominal*peak_radius*sigma*res*delta_peak_density
             
       
        delta_wlf_h_real = (1/(area_real*((1/(sp/1000))-(1/(sr/1000)))))*deltam
        
        delta_wlf_area_real = (h/(-(area_real**2)*((1/(sp/1000))-(1/(sr/1000)))))*delta_area_real
        # delta_wlf_area_real = (height/(-(real_area**2)*((1/(probensteigung/1000))-(1/(referenzsteigung/1000)))))*delta_area_real
        delta_wlf_area_real_std = (h/(-(area_real**2)*((1/(sp/1000))-(1/(sr/1000)))))*std_area_real_10
        
        # delta_wlf_real_einzeln = abs(delta_wlf_h_real) + abs(delta_wlf_area_real)
        # delta_wlf_real_einzeln = abs(delta_wlf_h_real) + abs(delta_wlf_area_real)
        # delta_wlf_real_einzeln_std = abs(delta_wlf_h_real) + abs(delta_wlf_area_real_std)
        
        # delta_wlf_real = np.append(delta_wlf_real, np.mean(delta_wlf_real_einzeln))
        # delta_wlf_real = np.append(delta_wlf_real, np.mean(delta_wlf_real_einzeln))
        # delta_wlf_real_std = np.append(delta_wlf_real_std, np.mean(delta_wlf_real_einzeln_std))
    

    werte = pd.DataFrame({
        'Rauheit': rauheit,
        'Höhe': probenhöhe,
        'Durchmesser': durchmesser,
        'Nominal Area': nominal_area,
        'Real Area': real_area,
        'Probensteigung': probensteigung,
        'Referenzsteigung': referenzsteigung,
        'WLF Nominal': wlf_einzeln_nominal})

#%%----------------------------------------------------------------------------

wlfmitdelta_nominal = wlf_mean_nominal + delta_wlf_nominal
wlfmitstd_nominal = wlf_mean_nominal + std_mean_nominal

# wlfmitdelta_real = wlf_mean_real + delta_wlf_real_std
# wlfmitstd_real = wlf_mean_real + std_mean_real

#%%----------------------------------------------------------------------------

x_achse = np.array([1, 3, 5])
y_achse = np.array([0.79, 0.44, 0.1])

ziel = 3.68
faktor = 1/(0.77**2)

get_3dplot(x_achse, y_achse, wlf_mean_nominal, wlfmitdelta_nominal,
            'NOMINAL WLF mit Fehlerfortpflanzung')
# get_3dplot(x_achse, y_achse, wlf_mean_nominal, wlfmitstd_nominal,
#             'NOMINAL WLF mit STD')
get_3dplot_prozent(x_achse, y_achse, wlf_mean_nominal, wlfmitdelta_nominal, ziel, faktor,
                    'NOMINAL WLF mit Fehlerfortpflanzung proz. Abstand')
# get_3dplot_prozent(x_achse, y_achse, wlf_mean_nominal, wlfmitstd_nominal, 3.76,
#                     'NOMINAL WLF mit STD proz. Abstand')

# get_3dplot(x_achse, y_achse, wlf_mean_real, wlfmitdelta_real,
#            'REAL WLF mit Fehlerfortpflanzung')
# get_3dplot(x_achse, y_achse, wlf_mean_real, wlfmitstd_real,
#            'REAL WLF mit STD')

#%%----------------------------------------------------------------------------