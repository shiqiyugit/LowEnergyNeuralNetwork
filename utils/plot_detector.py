#template to plot event display

import matplotlib as plt
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import detector_rings

def detector_3d(ics, dcs, ic_vals, dc_vals):

  ICrings=detector_rings.create_ring_dict()
#  ICs = np.concatenate((ICrings[0],ICrings[1]))
#  ICs = np.concatenate((ICs,ICrings[2]))

  file_name = "/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/utils/detector_information/icecube_stringsXY.txt"
  print("Using file from %s"%file_name)

  string36 = "/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/utils/detector_information/icecube_string36.txt"
  string86 = "/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/utils/detector_information/icecube_string86.txt"

  XYs = np.genfromtxt(file_name)
  #x=rawdata[:,1]
  #y=rawdata[:,2]
  fz36= np.genfromtxt(string36)
  #ICs
  Xic=[]
  Yic=[]
  Zic=[]
  vals=[]
  zics=fz36[:,3]

  for ic in ics:
    xic = XYs[ic-1,1]
    xics = [xic for x in range(zics.shape[0])]
    Xic = np.concatenate((Xic,xics))
    yic = XYs[ic-1,2]
    yics = [yic for x in range(zics.shape[0])]
    Yic = np.concatenate((Yic, yics))
    Zic = np.concatenate((Zic,zics))
  fz86= np.genfromtxt(string86)
#DC
  zdcs=fz86[:,3]
#  for dc in range(78,86):
  for dc in dcs:
    xdc = XYs[dc-1,1]
    xdcs = [xdc for x in range(zdcs.shape[0])]
    Xic = np.concatenate((Xic,xdcs))
    ydc = XYs[dc-1,2]
    ydcs = [ydc for x in range(zdcs.shape[0])]
    Yic = np.concatenate((Yic, ydcs))
    Zic = np.concatenate((Zic,zdcs))
  vals = np.concatenate((ic_vals,dc_vals))
  return Xic, Yic, Zic, vals

#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Xic, Yic, Zic)

#plt.show()

#ax=fig.add_subplot(111,projection='3d')
#ax.
