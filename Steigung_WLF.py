# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:31:47 2022

@author: Simon
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


def read_unicode_txt(file, plot):
    b = np.array([])
    
    # open and read file as lines of string
    with open(file, 'r', encoding='utf-16') as f:
        content = f.readlines()[11:731]
    
    data_np = np.zeros((len(content), 5), dtype=object)
    'Colums: 0 - Index; 1 - time; 2 - temp_sample; 3 - temp_ref; 4 - heat_flow'
    
    
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
    d = {'index': data_np[:, 0], 'time': data_np[:,1], 'temp_sample': data_np[:,2], 'temp_ref': data_np[:,3],
         'heat_flow': data_np[:,4]}
    data = pd.DataFrame(data=d)
    
    if plot == True:
        plt.figure(figsize=[13,10])
        plt.plot(data['temp_sample'], data['heat_flow'])
        plt.title('Segment 6 - WLF-Messung', size=25, weight='bold')
        plt.xlabel('Probentemperatur t_sample [°C]', size=25)
        plt.ylabel('Heat Flow Q̇ [mW]', size=25)
        plt.grid('on', linestyle='--')
        plt.tick_params(labelsize = 25)
        plt.show()
    
    else:
        None
     
    return data
    


#%%----------------------------------------------------------------------------

file = select_file()



data = read_unicode_txt(file, True)

