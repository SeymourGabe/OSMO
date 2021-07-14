# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 17:33:40 2021

@author: chra3427
"""
import os,sys
sys.path.append('C:/Users/Admin.DESKCOMPUTER/AppData/Local/Programs/Python/Python38/Lib/site-packages')
#os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path
#sys.path.append('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path
import numpy as np
import scipy.optimize as opt
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb
from stl import mesh   
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import preProcessForTomo as pp4t
from scipy import fft
from scipy import ndimage
import copy
import setUpLTTModule as lttpy
OSMOdir = 'C:/Users/Admin.DESKCOMPUTER/Desktop/OSMO/'
LTTdir = 'C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python'
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/OSMO') #Add LTT python dll's to path
import OSMOmodule as osmo
from matplotlib.ticker import MaxNLocator
from matplotlib_scalebar.scalebar import ScaleBar
import datetime as dt



#  GTS: Get current datetime in YYYY-MM-DD_HH.MM. Create folder here
testDict = {}
dateTime = osmo.getDateString()
#print("dateTime = ", dateTime) 

savePath = os.path.join(OSMOdir, dateTime)
print("Saving variables to ", savePath, "\n") 

print("os.path.isdir(savePath) = ", os.path.isdir(savePath))
if (os.path.isdir(savePath)):
    print(savePath, " already exists. Saving to existing folder")

else:
    print(savePath, " does not exist. Creating folder")
    os.mkdir(savePath)



sb.set();               sb.set(font_scale=1.5)
sb.set_style("white");  mapStr    = 'CMRmap'
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path

from LTTserver import LTTserver
LTT                 = LTTserver()

# print('For users coming from Matlab, please note that:')
# print('x=1; y=x; y=y+1; Will then yeild x = 2. Use the copy.copy command to have variable names reference new memory.')
# print('Indexing includes the first term, but not the last. So A[0:5] will address the first elemnt of A, up to 4.')
# print('All volume data is in the form vol = np.array[z, x, y], where z = slices, [x,y] = [rows,cols].')
# print('g, the image set data, is in the form np.array[theta, x, y], where theta is the projection angle, [x,y] are DMD rows and columns.')
# print(' ')

# Input must be a 2D image (some images have more dimensions if they have rbg values defined)
# -------------     Import image and format it for LTT
# imOrig = imageio.imread(r"C:\Users\chra3427\Desktop\playingWithPython\orig.tif")  #hex mesh with some small voids
# imOrig = imageio.imread(r"C:\Users\chra3427\Desktop\playingWithPython\largeHexMesh.tif")  #hex mesh
# imOrig = imageio.imread(r"C:\Users\chra3427\Google Drive\---- Research ----\01 VAM via Python\i.tif")    #pinwheel
# imOrig = imageio.imread(r"C:\Users\chra3427\Desktop\playingWithPython\resChart.tif")    #resolution target
#imOrig = imageio.imread(r"C:\Users\chra3427\Desktop\playingWithPython\earth.tif")  #earth grayscale
# imOrig = imageio.imread(r"C:\Users\chra3427\Desktop\playingWithPython\ringUncompressed.tif") 
# imOrig3D = imOrig

# imOrig3D = np.asarray(np.load(r"D:\Voxelized 3D targets from Joe\LLNLBox200.npy"))
##imOrig3D = np.as(r"C:/Users/Admin.DESKCOMPUTER/Downloads/SAXS_Stripes.npy")

imOrig3D = np.asarray(np.load(r"C:/Users/Admin.DESKCOMPUTER/Downloads/SAXS_Stripes.npy"))
# imOrig3D = np.asarray(np.load(r"D:\Voxelized 3D targets from Joe\LLNLBox200.npy"))
# imOrig3D = np.load(r"D:\Voxelized 3D targets from Joe\thinker200.npy")
# imOrig3D = osmo.trimZeroSlices(imOrig3D)
imOrig3D = np.rot90(imOrig3D, k=1, axes =(0,2) ) 
heightOrig = imOrig3D.shape[0]



fT, nrays, indsImgEdge      = osmo.padTarget(imOrig3D);         #Assumes form [Z,X,Y] for imOrig3D
fTorig                      = copy.deepcopy(fT)


# Plot Slices of target geometry
ZslicesToPlot = np.array([fT.shape[0]//2]);  XslicesToPlot = np.array([fT.shape[1]//2]);   YslicesToPlot = np.array([int(fT.shape[2]*.76)]);
# osmo.plotVolSlices(fTorig, ZslicesToPlot, XslicesToPlot, YslicesToPlot, mapStr='bone',dpiToUse=500)



#=============================================================
#                   Projection Parameters  
#=============================================================
angularRange            = 360
nAngles                 = 360
  
alpha                   = 0.040312    #22mM I907
desiredPrintDiam_mm     = 12.9   #mm  Desired diam
# desiredPrintHeight_mm   = 7.899

print('Test interp')
print('Calculate total variation of target')



#=============================================================
#               Scaling & Export Parameters  
#=============================================================
opticalMag              = 14/12.1     #DMD is 12.1mm in height, over 1600 pixels
degTilt                 = 7           # Matches DMD tilt as of 3/19/2021. The slight keystoning is not considered here (compression). 
desiredDMDdiam_voxels, desiredDMDheight_voxels   = osmo.determineTargetExtrapedResolution_Diam(desiredPrintDiam_mm, fT, degTilt, opticalMag)




#=============================================================
#                     OSMO Parameters 
#=============================================================
Dch                 = .90         # in (Dcl,1].             Default: Dch = 0.9.  These are the OSMO critical dose values.
Dcl                 = .86         # in [0,Dch).             Default: Dcl = 0.85. 
smallestProjVal     = 0           #The clipping condition.  Default: 0. Only negative to model inhibition. 
smallestDMDgray     = 0           # Default = 0. Now OMSO doesn't allow gray projector image values below smallestDMDgray*(projector max (white)). 
                                  # ^This doesn't seem to work. Probably remvoe frome code.
Nitr                = 10000           #How many OSMO iterations? Usually 100 is a good bet for more simple structures. Best to set very high, 
                                  # watch convergence, and manually stop when good.
normEachSlice       = 1           #Default is 1. 1 = each slice normalized to its max per OSMO iteration. 0 = all slices normed to global max of all slices each itr.
normToChoice        = 'MinGel'    #How to scale final slices to each other     # OPTIONS: 'MinGel', 'MaxGel', 'MidTh', 'MeanGel'. Default: MinGel

print('print info on normeachslice, on norm choice, on nAngles')


nslices = fT.shape[0]                                                           # This gets used later for plotting
pxsize = desiredPrintDiam_mm/nrays
lttpy.setup_LTT3D(LTT,angularRange,nAngles,nrays,alpha, pxsize, nslices)


osmo.plotVolSlices_afterOpticalMag_withScaleBar(fT, pxsize, ZslicesToPlot, XslicesToPlot, YslicesToPlot,mapStr='bone',dpiToUse=500, scaleBarLengthFraction = .3,scalebarLocation = 'upper left'\
, scaleColor = 'darkorange', scaleBarAlpha = 0, barRotation = 'vertical', frameState = False, darkBackground = False)


voidInds, gelInds   = osmo.get_void_gel_indices(fTorig,LTT)                     # Where is in-part (gel), and out-of-part (void)?
fT_LTTfiltered      = osmo.apply2DrampFilter(fT,LTT)                            # Apply a smoothed version of a 2D Ram-Lak filter

#--------         OSMO Initilization     --------
fm              = fT_LTTfiltered   
#fm              = np.load(r"D:\SomeSavedArrays\6_9_2021 Stripes for Gabe\fm.npy") 
#print('-------------------------------------------------------------------------------')
#print('----------------------- Model loaded from previous run! -----------------------')
#print('-------------------------------------------------------------------------------')                  
f               = osmo.reconstructFromModel(fm,smallestProjVal,LTT,normEachSlice,smallestDMDgray)

#--------     Initialize VER and PW Arrays     -------
costVals        = [];  costVals         = np.append(costVals, [osmo.costFxn(f,voidInds,gelInds)], axis=0)
processWindows  = [];  pw               = osmo.processWindow(0,f,voidInds,gelInds)[0]; 
processWindows  = np.append(processWindows, [pw], axis=0)
#costVals        = np.load(r"D:\SomeSavedArrays\6_9_2021 Stripes for Gabe\costVals.npy") 
#processWindows  = np.load(r"D:\SomeSavedArrays\6_9_2021 Stripes for Gabe\processWindows.npy")


#=======================================================================================================
#=======================================================================================================  
ii = 0;  run = 1 #                          OSMO LOOP  
#==============================================================================
while run == 1: 
    ii = ii + 1     # First value should be ii = 1, since we already did the 0th initialization step, and saved PW
    if ii == Nitr:
        run = 0
        
    f, fm           = osmo.iterateOSMO(Dcl,Dch, voidInds,gelInds, f,fm, smallestProjVal,LTT, normEachSlice,smallestDMDgray)
    
    costVals        = np.append(costVals, [osmo.costFxn(f,voidInds,gelInds)], axis=0)
    pw              = osmo.processWindow(0,f,voidInds,gelInds)[0];      
    processWindows  = np.append(processWindows, [pw], axis=0)
    print(str(ii)+'. VER = '+str(costVals[-1])+'. PW = '+str(processWindows[-1]))
     
    if np.mod(ii,100)==0:
        plt.plot(dpi = 500);           ax = plt.plot(processWindows,linewidth=3);
        plt.title('Process Window');   plt.xlabel('Iterations');    plt.ylabel('Normalized Dose')
        plt.show();    
        
    if np.mod(ii, 10)==0 or ii == 1:
        osmo.saveManyVars(fm, f, voidInds, gelInds, costVals, pw, ii, savePath)
        
        
#==============================================================================
fRaw = copy.copy(f);    fmRaw = copy.copy(fm);  
osmo.plotDoseHists(fRaw,voidInds,gelInds, titleStr = 'Voxel Doses before g-scaling', dpiToUse=500)
# normToChoice        = 'MinGel'

f = copy.copy(fRaw);    fm = copy.copy(fmRaw);  

# Rescale projection images, then reconstruct.  Not part of OSMO loop, but critical since slices were optimized independently (or at least mostly so)
# This makes is so that all projector image rows are scaled such that, when backprojected, all slices of f have the same units of dose        
g, f, maxVoidPerSlice, minGelPerSlice, maxGelPerSlice  = osmo.scale_all_slices(fm, voidInds,gelInds, smallestProjVal, LTT, smallestDMDgray, normTo = normToChoice)    

pw      = osmo.processWindow(0,f,voidInds,gelInds)[0];     print('Final PW = ' + str(pw))  # Get the process window after scaling. It'll be >= PW size before scaling
#=======================================================================================================  
#=======================================================================================================  


    

#=======================================================================================================
#                            Make Plots of Reconstrucion and OMSO Convergence
#=======================================================================================================
#Plot Slices of reconstruction
ZslicesToPlot = np.array([f.shape[0]//2]);  
XslicesToPlot = np.array([f.shape[1]//2]);   
YslicesToPlot = np.array([f.shape[2]//2]);
osmo.plotVolSlices(f, ZslicesToPlot, XslicesToPlot, YslicesToPlot, mapStr='CMRmap',dpiToUse=500)

#Plot projector images
anglesToPlot    = np.array([0,10]);     osmo.plot_ProjectionImages(g,anglesToPlot,mapStr='CMRmap',  minPltVal=0,maxPltVal=1, dpiToUse=200)

# --------   Plot VER   --------
plt.figure(dpi = 500);      ax = plt.plot(costVals,linewidth=3);      plt.title('Convergence');                   
plt.xlabel('Iterations');   plt.ylabel('VER');   plt.show()

# --------   Plot PW   --------
plt.figure(dpi = 500);      ax = plt.plot(processWindows,linewidth=3); plt.title('Process Window Convergence');   
plt.xlabel('Iterations');   plt.ylabel('PW');    plt.show()

#------   Plot Log Hists   -----
osmo.plotDoseHists(f,voidInds,gelInds, titleStr = 'Voxel Doses after g-scaling', dpiToUse=500)


#---------   Plot: [max in-part voxel dose] + [IMDR] per slice  --------
osmo.plotDoseRangesPerSlice(nslices, maxVoidPerSlice, minGelPerSlice, maxGelPerSlice, fm, LTT, useNormedImageSet = False, dpiToUse = 500)


#---------   Make plots to show mean SCALED voxel intensity per rotation, normed to max I --------
fScalingFactor = osmo.makeHist_avgIntensityPerVoxel(fm,voidInds,gelInds,LTT)


#---------    Plot: SCALED [max in-part voxel dose] + [IMDR] per slice ---------
osmo.plotDoseRangesPerSlice(nslices, maxVoidPerSlice, minGelPerSlice, maxGelPerSlice, fm, LTT, useNormedImageSet = True, dpiToUse = 500)


g = g+np.min(g)
g = g/np.max(g)

# --- Hist of image set ---
plt.figure(dpi = 1000)  
sb.set_style("white")
sb.distplot(np.ravel(g), color="black",kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
plt.title('Distribution of Projector Pixel Values\nDarkest Gray Allowed = ' + str(smallestDMDgray))
plt.ylabel('Pixel Counts');     plt.xlabel('Normalized Gray Values');     plt.show()






#=======================================================================================================
#                 Interpolate to DMD resolution, rotate, and save to image files
#=======================================================================================================

#### ------ Scale to desired size on DMD, and rotate -------
desiredDMDdiam_voxels, desiredDMDheight_voxels   = osmo.determineTargetExtrapedResolution_Diam(desiredPrintDiam_mm, fT, degTilt, opticalMag)
_, gResized_Rotated = osmo.interpOntoDMD(g, desiredDMDdiam_voxels, degTilt)

#### ------- Plot a few of the projection images that are about to be saved ----
anglesToPlot               = np.array([0,10]);     
osmo.plot_ProjectionImages(gResized_Rotated,    anglesToPlot, mapStr='CMRmap', minPltVal=0,maxPltVal=1, dpiToUse=200)

#### -------- Save images to file ----------
osmo.saveImageSetWithoutRotation(gResized_Rotated,r"D:\Saved Image Sets from Python", degTilt)
print('Images Saved To Files')







# Make this a plotting function (specify slice, then line thorugh that slice)
# shiftFromCenter = -10
# sliceAxis       = np.arange(0,nslices)
# plt.figure(dpi = 500);      ax = plt.plot(f[:,f.shape[1]//2,  f.shape[1]//2 + shiftFromCenter],sliceAxis, linewidth=3);    
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.gca().invert_yaxis();        
# plt.xlabel('Relative Dose');     plt.title('Dose through [:,Nx//2, Ny//2'+str(shiftFromCenter)+']');  plt.ylabel('Slice #');      
# plt.show()

# plt.figure(dpi = 500)
# ax = sb.heatmap(f[:,f.shape[1]//2,:],cmap='bone')
# ax.set_aspect('equal')
# plt.title('X-Slice #'+str(f.shape[1]//2))
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# plt.axvline(x=f.shape[1]//2+shiftFromCenter, ymin=0, ymax=1,color='r',alpha = 0.5, linestyle = '--'); 
# plt.show() 























# # #------------------ Plot hists, and overplot dose ranges per slice -------------
# for ii in range(maxVoidPerSlice.shape[0]):
#     if maxGelPerSlice[ii] == 0:
#         maxVoidPerSlice[ii] = np.nan
#         minGelPerSlice[ii] = np.nan
#         maxGelPerSlice[ii] = np.nan
# fig, ax = plt.subplots(dpi = 500)
# sb.distplot(np.ravel(f[voidInds[0],voidInds[1],voidInds[2]]),       color="red",  label="Out-of-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True,'alpha':0.4},ax=ax)
# sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]),          color="navy", label="In-Part",    kde=False,norm_hist = False,hist = True,hist_kws={'log':True,'alpha':.2},ax=ax)
# plt.xlabel('Normalized Dose');      plt.ylabel('Voxel Counts');     plt.title('In-Part Voxel Doses [Hists] \nMin Dose Per Slice [Markers]')
# ax2 = ax.twinx()
# sliceAxis = np.arange(0,fm.shape[0])
# for ii in range(fm.shape[0]):
#     plt.plot([ minGelPerSlice[ii],maxGelPerSlice[ii] ],[sliceAxis[ii],sliceAxis[ii]],color='darkslateblue', alpha = .4)
#     plt.plot(maxVoidPerSlice[ii],sliceAxis[ii],color = 'firebrick', linestyle="",marker=".",alpha = 0.4)
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.gca().invert_yaxis();        plt.ylabel('Slice #');       plt.show()



# col1 = "darkslateblue"
# col2 = "firebrick"
# fig, ax = plt.subplots(dpi = 500)
# sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]), color="navy", label="In-Part",kde=False, norm_hist = False,hist = True,hist_kws={'log':True,'alpha':0.5},ax=ax)
# plt.xlabel('Normalized Dose')
# plt.ylabel('Voxel Counts')
# plt.title('Hist: In-Part Voxel Doses \nPoints: Min Dose Per Slice')
# ax.tick_params(axis='y', colors=col1)
# ax.tick_params(axis='y', colors=col1)
# ax.yaxis.label.set_color(col1)
# ###########
# ax2 = ax.twinx()
# plt.plot(minGelPerSlice,sliceAxis,color = col2, linestyle="",marker="*",alpha = 0.8)
# plt.gca().invert_yaxis()
# plt.ylabel('Slice #')
# ax2.tick_params(axis='y', colors=col2)
# ax2.tick_params(axis='y', colors=col2)
# ax2.yaxis.label.set_color(col2)
# plt.tight_layout()
# plt.show()


# This one pretty good. 
# fig, ax = plt.subplots(dpi = 500)
# sb.distplot(np.ravel(f[voidInds[0],voidInds[1],voidInds[2]]), color="red", label="Out-of-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True,'alpha':0.4},ax=ax)
# sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]), color="navy", label="In-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True,'alpha':0.5},ax=ax)
# plt.xlabel('Normalized Dose')
# plt.ylabel('Voxel Counts [Hist]')
# plt.title('In-Part Voxel Doses [Hists] \nMin Dose Per Slice [Markers]')
# ###########
# ax2 = ax.twinx()
# plt.plot(minGelPerSlice,sliceAxis,color = 'navy', linestyle="",marker="*",alpha = 0.5)
# plt.plot(maxVoidPerSlice,sliceAxis,color = 'darkred', linestyle="",marker="*",alpha = 0.5)
# plt.gca().invert_yaxis()
# plt.ylabel('Slice # [Markers]')
# plt.tight_layout()
# plt.show()















