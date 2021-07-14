# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:29:28 2021

@author: Charles Rackson

Module for VAM Optimization with the OSMO algorithm.

"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import setUpLTTModule as lttpy
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/OSMO') #Add LTT python dll's to path
import preProcessForTomo as pp4t
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path
import copy
import cv2
from PIL import Image
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
import datetime as dt
from dataclasses import dataclass



#os.chdir('C:/LTT/v1.6.10/python') #Add LTT python dll's to path
#os.chdir('C:/LTT/LTT_v1.6.25/python') #Add LTT python dll's to path
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path

from LTTserver import LTTserver
LTT = LTTserver()

# ----------- Housekeeping functions ---------------
def getDateString():
    """
    GTS: This function returns ta date string in the format 
    YYYY-MM-DD_HH.MM. Used to create a folder to save vars to. 
    """
    timeStr = str(dt.datetime.today())
    timeLen = len(timeStr)
    timeStrSv = timeStr[:timeLen-10]
    timeStrSv = timeStrSv.replace(" ", "_")
    timeStrSv = timeStrSv.replace(":", ".")
    return timeStrSv

def saveVar(var, varName, folder):
    """
    GTS: This function takes a variable and path. Saves the variable to the path
    """
   # fldr = "C:/Users/Admin.DESKCOMPUTER/Desktop/OSMO/test"
    saveTo = folder + "/" + varName
    print("saveTo = ", saveTo)
    np.save(saveTo, var)
    print("saved")
    
def saveManyVars(fm, f, voidInds, gelInds, costVals, pw, ii, savePath):
    """
    GTS: This function uses osmo.saveVar to save predetermined variables to 
    savePath folder.
    """
    print("in saveManyVars")
    saveVar(fm, "fm", savePath)
    saveVar(f, "f", savePath)
    saveVar(voidInds, "voidInds", savePath)
    saveVar(gelInds, "gelInds", savePath)
    saveVar(costVals, "costVals", savePath)
    saveVar(pw, "pw", savePath)
    saveVar(ii, "ii", savePath)
    print("finsihing saveManyVars")
    
# ----------- Pre processing functions ---------------
def trimZeroSlices(f):
    # Takes a 3D array [z,x,y]
    # Returns the array with any extra zero-padding removed.
    # Zero slices piercing the object are not removed, only surrounding zero slices. 
    # Surrounding zero slices in all 3 dims are removed.
    # E.g. extra zero slices top and bottom, left and right, fore and aft. 
    
    # First get z ind limits of nonzero regions
    trimTop = True;     ii = 0
    while trimTop == True:
        if np.any(f[ii,:,:]):            # If this slice is nonzero
            zIndLow = ii
            trimTop = False
        ii = ii+1
    
    trimBottom = True;  ii = -1
    while trimBottom == True:
        if  np.any(f[ii,:,:]):            # If this slice is nonzero
            zIndHigh = ii
            trimBottom = False
        ii = ii-1  

    # Next get x lims
    trimTop = True;     ii = 0
    while trimTop == True:
        if np.any(f[:,ii,:]):            
            xIndLow = ii
            trimTop = False
        ii = ii+1
    
    trimBottom = True;  ii = -1
    while trimBottom == True:
        if np.any(f[:,ii,:]):        
            xIndHigh = ii
            trimBottom = False
        ii = ii-1

    
    # Next trim in y
    trimTop = True;     ii = 0
    while trimTop == True:
        if np.any(f[:,:,ii]):      
            yIndLow = ii
            trimTop = False
        ii = ii+1
    
    trimBottom = True;  ii = -1
    while trimBottom == True:
        if np.any(f[:,:,ii]):     
            yIndHigh = ii
            trimBottom = False
        ii = ii-1
        
    f = f[zIndLow:zIndHigh, xIndLow:xIndHigh, yIndLow:yIndHigh]    
    
    return f

def padTarget(imOrig3D):
    #Assumes form [Z,X,Y] for imOrig3D
    Ndims = np.size(imOrig3D.shape)
    
    if Ndims == 2: 
        imOrig3D = np.expand_dims(imOrig3D, axis=0)

        fT0, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[0,:,:], makeBinary=True)
        fT                      = np.float32(fT0)
    
    if Ndims > 2:   
        fT0, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[0,:,:], makeBinary=True)
        fT                      = np.float32(np.zeros([imOrig3D.shape[0],fT0.shape[1],fT0.shape[2]]))
        for ss in range(fT.shape[0]):    
            fTss, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[ss,:,:], makeBinary=True)
            fT[ss,:,:] = fTss[0,:,:]
    
    
    fT = np.float32(fT);    fT[np.isnan(fT)]=0    
    
    return fT, nrays, indsImgEdge




def padTarget_tomosynthesis(imOrig3D, tiltAngle):
    # Tilt angle is the tomosnythesis tilt angle (zero = tomography) in degrees
    
    #Assumes form [Z,X,Y] for imOrig3D
    Ndims = np.size(imOrig3D.shape)
    
    # First pad around axis of rotation; normal tomography padding
    if Ndims == 2: 
        imOrig3D = np.expand_dims(imOrig3D, axis=0)

        fT0, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[0,:,:], makeBinary=True)
        fT                      = np.float32(fT0)
    
    if Ndims > 2:   
        fT0, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[0,:,:], makeBinary=True)
        fT                      = np.float32(np.zeros([imOrig3D.shape[0],fT0.shape[1],fT0.shape[2]]))
        for ss in range(fT.shape[0]):    
            fTss, nrays, indsImgEdge = pp4t.prepIm4LTT(imOrig3D[ss,:,:], makeBinary=True)
            fT[ss,:,:] = fTss[0,:,:]
    
    
    
    # Next, pad vertically by adding the correct number of zero slices up and down to make it to that tilting obj away from viewer does not clip top and bottom
    h = fT.shape[0]                       #Height before vertical padding
    d = np.max([fT.shape[1],fT.shape[2]])
    
    H = np.abs(d*np.sin(tiltAngle*np.pi/180)) + np.abs(h*np.cos(tiltAngle*np.pi/180))          #Final desired height after vertical padding
    
    delH = H-h
    
    if np.mod(delH,2)==0: # If num z slices to add is even
        toAddTop    = int(delH/2)
        toAddBottom = int(delH/2)
    else:
        toAddTop    = int(delH//2+1)
        toAddBottom = int(delH//2)
        
    fT  = np.pad(fT,[(toAddTop,toAddBottom), (0,0), (0,0)])
    # Done vertically padding
    
    
    fT = np.float32(fT);    fT[np.isnan(fT)]=0    
    
    return fT, nrays, indsImgEdge



# ----------- Functions used for OSMO optimization: -------------
def iterateOSMO(Dcl,Dch, voidInds,gelInds, f,fm, smallestProjVal,LTT, normEachSlice=0, smallestDMDgray= 0):
    # Takes a model (fm) and a reconstruction (f), and iterates OSMO once. Outputs fm and f.
    # Parameters like alpha, Nangles, etc, are set in another function via LTT
    
    # if normEachSlice = 0, upon backprojection, the entire reconstrucion volume is normalized to its max (even though slices aren't coupled)
    # if normEachSlice = 1, each slice is normalized to its own maximum
    
    
    ## FIRST REDUCE VOID DOSE##################################################
    ##
    ##
    # ============= Subtract values from void voxels in model ===============
    voidDif1 = f[voidInds[0],voidInds[1],voidInds[2]]-Dcl   #In the voids, how much over the critical dose we are
    voidDif1[voidDif1<0]=0                                  #If a void reconstruction point already below Dcl, don't do anything to it
    # Construct a new target, with some negatives added
    fm[voidInds[0],voidInds[1],voidInds[2]] = fm[voidInds[0],voidInds[1],voidInds[2]]-voidDif1

    # Get new recontruction:
    # Project modified target, and set any negatives to zero
    g = lttpy.project(fm,LTT)
    if smallestProjVal != 0:
        g = g/np.max(g)    
    g[g<smallestProjVal]=smallestProjVal
    
    if smallestDMDgray != 0:
        # lowClip = smallestDMDgray*(np.max(g)-np.min(g)) - np.min(g)
        # g[g<lowClip]=lowClip
        g = scaleGtoLimitedDMDgrayRange(g,smallestDMDgray)
    
    # ================   Get new reconstruction   ===================
    f = lttpy.backProject(g,LTT)
    if normEachSlice:               
        for ii in range(f.shape[0]):
            m = np.max(f[ii,:,:])
            if m != 0:
                f[ii,:,:] = f[ii,:,:]/m
    else:
        f = f/np.max(f)
    f[f<0]=0   # Done reducing void dose
    
    
    
    
    ## NEXT INCREASE IN-PART(GEL) DOSE#########################################
    ##
    ##
    # ============= ADD values to gel voxels in model ===============
    gelDif = Dch - f[gelInds[0],gelInds[1],gelInds[2]]  #How far below Dch in the gel regions are we currently?
    gelDif[gelDif<0]=0  #If we're alrady above Dch in a gel region, don't try to increase dose there
    
    fm[gelInds[0],gelInds[1],gelInds[2]] = fm[gelInds[0],gelInds[1],gelInds[2]] + gelDif  #Add that ammount of dose to target
    
    # Project modified target, and set any negatives to zero
    g = lttpy.project(fm,LTT)
    if smallestProjVal != 0:
        g = g/np.max(g)
    g[g<smallestProjVal]=smallestProjVal
    
    if smallestDMDgray != 0:
        # lowClip = smallestDMDgray*(np.max(g)-np.min(g)) - np.min(g)
        # g[g<lowClip]=lowClip
        g = scaleGtoLimitedDMDgrayRange(g,smallestDMDgray)
    
    # ================   Get new reconstruction   ===================
    f = lttpy.backProject(g,LTT)
    
    if normEachSlice:
        for ii in range(f.shape[0]):
            m = np.max(f[ii,:,:])
            if m != 0:
                f[ii,:,:] = f[ii,:,:]/m
    else:
        f = f/np.max(f)
    f[f<0]=0   # Done increaseing in-part(gel) dose

    return f, fm

def scaleGtoLimitedDMDgrayRange(g,smallestDMDgray):
    
    if smallestDMDgray != 0:
        a = np.min(g);  b = np.max(g); 
        
        if a>=0:
            c = smallestDMDgray*(b-a) + a
            g = g* ((b-c)/(b-a))
            g = g + (b - np.max(g))
        elif a<0:
            a2 = a + np.abs(a); b2 = b + np.abs(a);
            c2 = smallestDMDgray*(b2-a2) + a2
            g = g* ((b2-c2)/(b2-a2))
            g = g + (b2 - np.max(g)) - np.abs(a)
    
    return g

def costFxn(f,voidInds,gelInds):
    smallestGelDose     = np.min(f[gelInds[0],gelInds[1],gelInds[2]])
    voidDoses           = f[voidInds[0],voidInds[1],voidInds[2]]
    nPixOverlap         = voidDoses[voidDoses>=smallestGelDose].size
    PER                 = nPixOverlap/(gelInds.shape[1] + voidInds.shape[1])
    return PER

def processWindow(allowedPER,f,voidInds,gelInds):
    Nvoxels             = gelInds.shape[1]+voidInds.shape[1]
    NerrPixelsAllowed = np.round(Nvoxels*allowedPER)
    

    
    if allowedPER==0:
        voidCutOffDose  = np.max(f[voidInds[0],voidInds[1],voidInds[2]])
        gelCutOffDose   = np.min(f[gelInds[0],gelInds[1],gelInds[2]])
        maxGelDose      = np.max(f[gelInds[0],gelInds[1],gelInds[2]])
    else:
        ascendingGelDoses   = np.ravel(f[gelInds[0],gelInds[1],gelInds[2]])
        ascendingGelDoses.sort()
        ascendingVoidDoses  = np.ravel(f[voidInds[0],voidInds[1],voidInds[2]])
        ascendingVoidDoses.sort()
        
        voidCutOffDose = ascendingVoidDoses[int(-1*np.floor(NerrPixelsAllowed))]
        gelCutOffDose = ascendingGelDoses[int(np.ceil(NerrPixelsAllowed))]
        maxGelDose = ascendingGelDoses[-1]

    windowSize = gelCutOffDose-voidCutOffDose
    minGelDose = gelCutOffDose
    
    maxVoidDose = voidCutOffDose
    return windowSize, minGelDose, maxVoidDose, maxGelDose

def findBestThreshold(f,threshesToTry,voidInds,gelInds):
    
    largestVoidDose     = np.max(f[voidInds[0],voidInds[1],voidInds[2]])
    smallestGelDose     = np.min(f[gelInds[0],gelInds[1],gelInds[2]])
    
    if largestVoidDose < smallestGelDose:
        bestThVal = 0.5*(largestVoidDose+smallestGelDose)
        
    else:
            
        score = np.zeros([1,np.size(threshesToTry)])
        
        for kk in range(np.size(threshesToTry)):
            th = threshesToTry[kk]
            
            threshed = copy.deepcopy(f)
            threshed[threshed<th] = 0
            threshed[threshed>=th] = 1
            
            numGelledInVoid = np.sum(threshed[voidInds[0],voidInds[1],voidInds[2]])
            numGelledInPrint = np.sum(threshed[gelInds[0],gelInds[1],gelInds[2]])
    
            score[0,kk] = numGelledInPrint/np.shape(gelInds)[1] - numGelledInVoid/np.shape(voidInds)[1]
            
    
        bestIndex = np.argmax(score)
        bestThVal = threshesToTry[bestIndex]
    
    return bestThVal

def reconstructFromModel(fm,smallestProjVal,LTT,normEachSlice=0, smallestDMDgray= 0):
    # Porject model: 
    g = lttpy.project(fm,LTT)
    g = g/np.max(g)
    g[g<smallestProjVal]=smallestProjVal
    
    if smallestDMDgray != 0:
        # lowClip = smallestDMDgray*(np.max(g)-np.min(g)) - np.min(g)
        # g[g<lowClip]=lowClip
        g = scaleGtoLimitedDMDgrayRange(g,smallestDMDgray)
    
    # Backproject
    f = lttpy.backProject(g,LTT)
    
    # Normalize the reconstruction
    if normEachSlice == 1:
        for ii in range(f.shape[0]):     # For each slice
            m = np.max(f[ii,:,:])
            if m != 0:
                f[ii,:,:] = f[ii,:,:]/m
    else:
        f = f/np.max(f)
    
    f[f<0]=0
    return f

def get_void_gel_indices(fTorig,LTT):
    
    g0 = lttpy.project(fTorig,LTT);               #Get projections    
    gOnes           = copy.copy(g0);
    gOnes[:,:,:]    = 1;
    RaStar1         = lttpy.backProject(gOnes,LTT)
    
    gelInds = np.array(np.nonzero(fTorig))         #Indicies of the target geometry
    voidsSetPositive = RaStar1*np.abs(fTorig-1)  
    voidInds =   np.array(np.nonzero(voidsSetPositive))  #INdicies of voids, within the total print area
    
    return voidInds, gelInds

def getSlice_gel_void_inds(sliceNumber, voidInds, gelInds):
    
    voidTuples  = np.transpose(voidInds)
    gelTuples   = np.transpose(gelInds)
    
    voidIndsInSlice = np.transpose( voidTuples[voidTuples[:,0]==sliceNumber,:] )
    gelIndsInSlice  = np.transpose( gelTuples[gelTuples[:,0]==sliceNumber,:] )
        
    return voidIndsInSlice, gelIndsInSlice

def apply2DrampFilter(fT,LTT):
    LTT.setAllReconSlicesZ(fT);     
    LTT.cmd('filterVolume RAMP');
    fT_LTTfiltered = LTT.getAllReconSlicesZ();
    return fT_LTTfiltered


# ----------- Functions used for scaling output, plotting, etc. ----------
def scale_all_slices(fm, voidInds,gelInds, smallestProjVal, LTT, smallestDMDgray= 0, normTo = 'MinGel'):
    
    
    g           = lttpy.project(fm,LTT)
    gMax        = np.max(g)
    g[g<smallestProjVal*gMax]=smallestProjVal*gMax
    
    if smallestDMDgray != 0:
        # lowClip = smallestDMDgray*(np.max(g)-np.min(g)) - np.min(g)
        # g[g<lowClip]=lowClip
        g = scaleGtoLimitedDMDgrayRange(g,smallestDMDgray)
    
    f           = lttpy.backProject(g,LTT)
    f           = f/np.max(f)
    f[f<0]      = 0
    gScaled     = copy.copy(g)
    
    minGelPerSlice  = []
    maxGelPerSlice  = []
    maxVoidPerSlice = []
    
    for ii in range(f.shape[0]):  # For each slice
        if np.max(f[ii,:,:]) != 0:
            voidIndsInSlice, gelIndsInSlice     = getSlice_gel_void_inds(ii, voidInds, gelInds)
            
            minGelDose                          = np.min(f[gelIndsInSlice[0],gelIndsInSlice[1],gelIndsInSlice[2]])
            minGelPerSlice                      = np.append(minGelPerSlice, [minGelDose], axis=0)
            
            maxGelDose                          = np.max(f[gelIndsInSlice[0],gelIndsInSlice[1],gelIndsInSlice[2]])
            maxGelPerSlice                      = np.append(maxGelPerSlice, [maxGelDose], axis=0)
            
            maxVoidDose                         = np.max(f[voidIndsInSlice[0],voidIndsInSlice[1],voidIndsInSlice[2]])
            maxVoidPerSlice                     = np.append(maxVoidPerSlice, [maxVoidDose], axis=0)
            
            midThPerSlice                       = 0.5*(maxVoidPerSlice + minGelPerSlice)
            meanGelPerSlice                     = 0.5*(minGelPerSlice + maxGelPerSlice)


    if normTo == 'MinGel':
        jj = 0
        for ii in range(f.shape[0]):  # For each slice, normalize to min of gel dose (so all slices finish printing at same time)
            if np.max(f[ii,:,:]) != 0:
                gScaled[:,ii,:] = gScaled[:,ii,:] * np.max(minGelPerSlice)/minGelPerSlice[jj]
                jj = jj+1
            
    elif normTo == 'MaxGel':
        jj = 0
        for ii in range(f.shape[0]):  # For each slice, normalize to max of gel dose (All slices start to gel simultaniously)
            if np.max(f[ii,:,:]) != 0:
                gScaled[:,ii,:] = gScaled[:,ii,:] * np.max(maxGelPerSlice)/maxGelPerSlice[jj]
                jj = jj+1
        
    elif normTo == 'MidTh': 
        jj = 0
        for ii in range(f.shape[0]):  # For each slice, normalize to mean of max void dose and min gel dose. Thus th farthest from both for all slices.
            if np.max(f[ii,:,:]) != 0:
                gScaled[:,ii,:] = gScaled[:,ii,:] * np.min(midThPerSlice)/midThPerSlice[jj]
                jj = jj+1
    
    elif normTo == 'MeanGel':
        jj = 0
        for ii in range(f.shape[0]):  # For each slice, normalize to mean of max void dose and min gel dose. Thus th farthest from both for all slices.
            if np.max(f[ii,:,:]) != 0:
                gScaled[:,ii,:] = gScaled[:,ii,:] * np.min(meanGelPerSlice)/meanGelPerSlice[jj]
                jj = jj+1
    

        
    gScaled = gScaled/np.max(gScaled)                # Normalize all projection images to 1 for export and dmd intensity computation
    if smallestDMDgray != 0:
        # lowClip = smallestDMDgray*(np.max(gScaled)-np.min(gScaled)) - np.min(gScaled)
        # gScaled[gScaled<lowClip]=lowClip
        g = scaleGtoLimitedDMDgrayRange(g,smallestDMDgray)
    
    f               = lttpy.backProject(gScaled,LTT) # All slices of f have the correct relative magntudes. 
    f               = f/np.max(f)
    f[f<0]          = 0
    
    
    minGelPerSlice  = []
    maxGelPerSlice  = []
    maxVoidPerSlice = []
    for ii in range(f.shape[0]):  # For each slice
        if np.max(f[ii,:,:]) != 0:
            voidIndsInSlice, gelIndsInSlice   = getSlice_gel_void_inds(ii, voidInds, gelInds)
            minGelDose          = np.min(f[gelIndsInSlice[0],gelIndsInSlice[1],gelIndsInSlice[2]])
            minGelPerSlice      = np.append(minGelPerSlice, [minGelDose], axis=0)
        
            maxGelDose          = np.max(f[gelIndsInSlice[0],gelIndsInSlice[1],gelIndsInSlice[2]])
            maxGelPerSlice      = np.append(maxGelPerSlice, [maxGelDose], axis=0)
            
            maxVoidDose         = np.max(f[voidIndsInSlice[0],voidIndsInSlice[1],voidIndsInSlice[2]])
            maxVoidPerSlice     = np.append(maxVoidPerSlice, [maxVoidDose], axis=0)
        else:
            minGelPerSlice      = np.append(minGelPerSlice, [0], axis=0)
            maxGelPerSlice      = np.append(maxGelPerSlice, [0], axis=0)
            maxVoidPerSlice     = np.append(maxVoidPerSlice, [0], axis=0)
    
    
    if smallestProjVal < 0:
        # gScaled = gScaled + np.min(gScaled)
        print('Image set has neatives. Shift to non-negative (call it g_shifted) for export')
        print('f is not dose - it is actually normed conversion. Back Project g_shifted to get actual applied dose')
    
    return gScaled, f,   maxVoidPerSlice, minGelPerSlice, maxGelPerSlice

def plot_ProjectionImages(g,anglesToPlot,mapStr='gray', minPltVal=0,maxPltVal=1, dpiToUse=500):
    
    for ii in range(anglesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap( g[anglesToPlot[ii],:,:],cmap=mapStr,vmin=minPltVal, vmax=maxPltVal)
        plt.title('Projection Image: '+str(anglesToPlot[ii])+'$^\circ$')
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()
        
    return

def plotVolSlices(f, ZslicesToPlot, XslicesToPlot, YslicesToPlot,mapStr='CMRmap',dpiToUse=500):
    
    for ii in range(ZslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[ZslicesToPlot[ii],:,:],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('Z-Slice #'+str(ZslicesToPlot[ii]))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
    for ii in range(XslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[:,XslicesToPlot[ii],:],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('X-Slice #'+str(XslicesToPlot[ii]))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
    for ii in range(YslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[:,:,YslicesToPlot[ii]],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('Y-Slice #'+str(YslicesToPlot[ii]))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
    return

def plotDoseHists(f,voidInds,gelInds, titleStr = 'Dose Hists',dpiToUse=500):
    plt.figure(dpi = 800);  sb.set_style("white")
    sb.distplot(np.ravel(f[voidInds[0],voidInds[1],voidInds[2]]),   color="red",  label="Out-of-Part", kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
    sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]),      color="navy", label="In-Part",     kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
    plt.legend(loc ="upper left");  plt.title(titleStr);   plt.xlabel('Normed Dose');    plt.ylabel('Counts');     plt.show()
    return

def plotDoseRangesPerSlice(nslices, maxVoidPerSlice, minGelPerSlice, maxGelPerSlice, fm, LTT, useNormedImageSet = False, dpiToUse = 500):
    
    for ii in range(maxVoidPerSlice.shape[0]):
        if maxGelPerSlice[ii] == 0:
            maxVoidPerSlice[ii] = np.nan
            minGelPerSlice[ii] = np.nan
            maxGelPerSlice[ii] = np.nan
    
    sliceAxis       = np.arange(0,nslices)
    
    
    if useNormedImageSet == False:       
        plt.figure(dpi = dpiToUse)
        for ii in range(nslices):
            plt.plot([ minGelPerSlice[ii],maxGelPerSlice[ii] ],[sliceAxis[ii],sliceAxis[ii]],color='darkslateblue', alpha = .9, label="IPDR")
            plt.plot(maxVoidPerSlice[ii],sliceAxis[ii],color = 'firebrick', linestyle="",marker=".",alpha = .9, label = "Max Out-of-Part Dose")
            
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().invert_yaxis();        
        plt.xlabel('Normalized Dose');     plt.title('Max Out-of-Part Doses\n& IPDRs Per Slice');  plt.ylabel('Slice #');      
        plt.show()
    
    
    
    elif useNormedImageSet == True:
        g       = lttpy.project(fm,LTT)
        g[g<0]  = 0
        g       = g/np.max(g)              # At print plane, highest intesity pixel out of entire set now corrosponds to a g value of 1
        f       = lttpy.backProject(g,LTT) # Value of each voxel is the avg intensity that voxels receives over one rotation, where units of intensity the same as g's units.
        fScalingFactor = np.max(f)
        
        minPrintDone = np.min(minGelPerSlice[~np.isnan(minGelPerSlice)])*fScalingFactor
        bestPrintDone = 0.5* (np.min(minGelPerSlice[~np.isnan(minGelPerSlice)]) + np.max(maxVoidPerSlice[~np.isnan(maxVoidPerSlice)])  )*fScalingFactor
        
        plt.figure(dpi = dpiToUse)
        for ii in range(nslices):
            plt.plot([ minGelPerSlice[ii]*fScalingFactor,maxGelPerSlice[ii]*fScalingFactor ],[sliceAxis[ii],sliceAxis[ii]],color='darkslateblue', alpha = .9, label="IPDR")
            plt.plot(maxVoidPerSlice[ii]*fScalingFactor,sliceAxis[ii],color = 'firebrick', linestyle="",marker=".",alpha = .9, label = "Max Out-of-Part Dose")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().invert_yaxis();    
            
        plt.axvline(x=minPrintDone, ymin=0, ymax=1,color='k',alpha = 0.5, linestyle = '--');   
        plt.axvline(x=bestPrintDone, ymin=0, ymax=1,color='g',alpha = 1, linestyle = '--');
        plt.xlabel('Avg. intensity on voxels.\nUnits: max I measured at print plane.');     
        plt.title('Avg. part-voxel normalized intensity received over 1 rotation.\n(Dose/rotation) = (these values)*(max print intensity)*(seconds/rotation)\n\nBest print (green line) @ ' + \
                  str(bestPrintDone)+'\nEarliest print @ '+str(minPrintDone) );  
        plt.ylabel('Slice #');       
        plt.show()
    
    
    return

def makeHist_avgIntensityPerVoxel(fm,voidInds,gelInds,LTT):
    # To compute effective intensity, as per reciprocity, scale gray values by actual intesnity/reciprocity result
    
    
    # g = getScaled_g_fromMap(fm,LTT)
    g       = lttpy.project(fm,LTT)
    g[g<0]  = 0
    g       = g/np.max(g)              # At print plane, highest intesity pixel out of entire set now corrosponds to a g value of 1
    
    f       = lttpy.backProject(g,LTT) # Value of each voxel is the avg intensity that voxels receives over one rotation, where units of intensity the same as g's units.
        
    plt.figure(dpi = 1000)  
    sb.set_style("white")
    sb.distplot(np.ravel(f[voidInds[0],voidInds[1],voidInds[2]]), color="red", label="Out-of-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
    sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]), color="navy", label="In-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
    plt.legend()
    plt.title('Avg. voxel normalized intensity received over 1 rotation.\n(Dose/rotation) = (these values)*(max masured intensity)*(time/rotation)')
    plt.ylabel('Voxel Counts')
    plt.xlabel('Avg. intensity on voxels.\nUnits: max I measured at print plane.')
    plt.show()
    
    plt.figure(dpi = 1000)  
    sb.set_style("white")
    sb.distplot(np.ravel(f[gelInds[0],gelInds[1],gelInds[2]]), color="navy", label="In-Part",kde=False,norm_hist = False,hist = True,hist_kws={'log':True})
    plt.legend()
    plt.title('Avg. part-voxel normalized intensity received over 1 rotation.\n(Dose/rotation) = (these values)*(max print intensity)*(seconds/rotation)')
    plt.show()
    
    fScalingFactor = np.max(f)
    return fScalingFactor

def plotVolSlices_afterOpticalMag_withScaleBar(f, pxsize, ZslicesToPlot, XslicesToPlot, YslicesToPlot,mapStr='CMRmap',dpiToUse=500, scaleBarLengthFraction = .2, \
                                               scalebarLocation = 'upper left',scaleColor = 'red', scaleBarAlpha = 0.5, barRotation = 'horizontal', frameState = 'False', darkBackground = False):

    # See https://pypi.org/project/matplotlib-scalebar/ for additional scalebar options
    if darkBackground:
        plt.style.use("dark_background")
    
    for ii in range(ZslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[ZslicesToPlot[ii],:,:],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('Z-Slice #'+str(ZslicesToPlot[ii]))
        scalebar = ScaleBar(pxsize, "mm", length_fraction=scaleBarLengthFraction,location=scalebarLocation,color = scaleColor,box_alpha = scaleBarAlpha, rotation=barRotation, frameon = frameState)
        ax.add_artist(scalebar)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
    for ii in range(XslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[:,XslicesToPlot[ii],:],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('X-Slice #'+str(XslicesToPlot[ii]))
        scalebar = ScaleBar(pxsize, "mm", length_fraction=scaleBarLengthFraction,location=scalebarLocation,color = scaleColor,box_alpha = scaleBarAlpha, rotation=barRotation, frameon = frameState)
        ax.add_artist(scalebar)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
    for ii in range(YslicesToPlot.size):
        plt.figure(dpi = dpiToUse)
        ax = sb.heatmap(f[:,:,YslicesToPlot[ii]],cmap=mapStr)
        ax.set_aspect('equal')
        plt.title('Y-Slice #'+str(YslicesToPlot[ii]))
        scalebar = ScaleBar(pxsize, "mm", length_fraction=scaleBarLengthFraction,location=scalebarLocation,color = scaleColor,box_alpha = scaleBarAlpha, rotation=barRotation, frameon = frameState)
        ax.add_artist(scalebar)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show() 
        
        # plt.style.use("seaborn")
        # sb.set();               sb.set(font_scale=1.5)
        # sb.set_style("white"); 
        # print(plt.style.available)    #This line prints out the list of plot styles
    return



#------------ Functions for rescaling, interpolating, and saving --------- 

def determineTargetExtrapedResolution_Diam(desiredPrintDiam_mm, fT, degTilt, opticalMag, DMD_Nrows = 1600, DMD_Ncols = 1600):
    # Returns the number of DMD pixels that should represent the diameter-range of the projection images
    # This number of pixels is BEFORE rotation.
    # So, after using this function, interpolate an image set to this width, THEN rotate it by degTilt, and
    # it should then both fit onto the 1600x1600 DMD after totation (change second if statement if more DMD columns illuminated),
    # and it should produce a print with diameter desiredPrintDiam_mm
    
    
    # NcolsDMD           = 2560
    # NrowsDMD           = 1600 #Resolution of VAM DMD. Vialux V9001
    
    # DMD_Nrows              = 1600
    pxSizeDMD              = 7.56E-3 #[mm]
    print(' ')
    print('Need to check actual height more precisely via pitch - ViaLux spec sheet slight mismatch')
    print('https://www.vialux.de/en/superspeed-specification.html')
    print('')
    print('Assuming a ' + str(DMD_Ncols)+'  x '+str(DMD_Nrows)+'  DMD because of illumination shape')

    desiredDMDDiam_mm      = desiredPrintDiam_mm / opticalMag
    desiredDMDHeight_mm    = desiredDMDDiam_mm* (fT.shape[0]/fT.shape[1])
    print(' ')
    print('A diameter of ' + str(desiredDMDDiam_mm) +'mm corrosponds to a print height of '+str(desiredDMDHeight_mm)+'mm')
    
    desiredDMDdiam_voxels   = int(np.round(desiredDMDDiam_mm/pxSizeDMD))
    desiredDMDheight_voxels = int(np.round(desiredDMDdiam_voxels*(fT.shape[0]/fT.shape[1])))
    
    Dmax_mm = DMD_Ncols*pxSizeDMD / ( np.sin(degTilt * np.pi/180) + (fT.shape[0]/fT.shape[1])*np.cos(degTilt * np.pi/180) )
    Hmax_mm = DMD_Nrows*pxSizeDMD / ( (fT.shape[1]/fT.shape[0])*np.cos(degTilt * np.pi/180) + np.sin(degTilt * np.pi/180) )

    Dmax_vox = Dmax_mm/pxSizeDMD
    Hmax_vox = Hmax_mm/pxSizeDMD
    

    
    # # Given the rotation, what's the widest the image should be before rotation?
    # Dmax_vox = DMD_Nrows / ( np.sin(degTilt * np.pi/180) + (fT.shape[0]/fT.shape[1])*np.cos(degTilt * np.pi/180) )
    # Hmax_vox = DMD_Nrows / ( (fT.shape[1]/fT.shape[0])*np.cos(degTilt * np.pi/180) + np.sin(degTilt * np.pi/180) )
    
    # Dmax_mm  = Dmax_vox*pxSizeDMD*opticalMag
    # Hmax_mm  = Hmax_vox*pxSizeDMD*opticalMag
    
    
    if desiredDMDDiam_mm > Dmax_vox:
        print('Error! Image width, corrosponding to requested print size, does not fit onto DMD width after rotation.')
        print(' ')
        print('Maximum Possible Print Diameter = ' + str(Dmax_mm) + ' mm')
        print('Maximum Possible Print Height = '   + str(Hmax_mm) + ' mm')
        print(' ')
        
        print('A 1600x1600 DMD is assumed, because of square illumination')
        sys.exit()
    
    if desiredDMDheight_voxels > Hmax_vox:
        print('Error! Image height, corrosponding to requested print size, does not fit onto DMD height after rotation.')
        print(' ')
        print('Maximum Possible Print Diameter = ' + str(Dmax_mm) + ' mm')
        print('Maximum Possible Print Height = '   + str(Hmax_mm) + ' mm')
        print(' ')
        print('A 1600x1600 DMD is assumed, because of square illumination')
        sys.exit()
    
    return desiredDMDdiam_voxels, desiredDMDheight_voxels


def determineTargetExtrapedResolution_tomosynthesis(desiredPrintDiam_mm, tiltAngle, fT, degTilt, opticalMag, LTT, DMD_Nrows = 1600, DMD_Ncols = 1600):
    # Returns the number of DMD pixels that should represent the diameter-range of the projection images
    # This number of pixels is BEFORE rotation.
    # So, after using this function, interpolate an image set to this width, THEN rotate it by degTilt, and
    # it should then both fit onto the 1600x1600 DMD after totation (change second if statement if more DMD columns illuminated),
    # and it should produce a print with diameter desiredPrintDiam_mm
    
    
    # NcolsDMD           = 2560
    # NrowsDMD           = 1600 #Resolution of VAM DMD. Vialux V9001
    
    # DMD_Nrows              = 1600
    pxSizeDMD              = 7.56E-3 #[mm]
    print(' ')
    print('Need to check actual height more precisely via pitch - ViaLux spec sheet slight mismatch')
    print('https://www.vialux.de/en/superspeed-specification.html')
    print('')
    print('Assuming a ' + str(DMD_Ncols)+'  x '+str(DMD_Nrows)+'  DMD because of illumination shape')

    desiredDMDDiam_mm      = desiredPrintDiam_mm / opticalMag
    desiredDMDHeight_mm    = desiredDMDDiam_mm* (fT.shape[0]/fT.shape[1])
    print(' ')
    print('A diameter of ' + str(desiredDMDDiam_mm) +'mm corrosponds to a print height of '+str(desiredDMDHeight_mm)+'mm')
    
    g                       = lttpy.project(fT,LTT)
    projHeight_voxels       = g.shape[1]
    projWidth_voxels        = g.shape[2]
    
    
    desiredDMDdiam_voxels   = int(np.round(desiredDMDDiam_mm/pxSizeDMD))
    desiredDMDheight_voxels = int(np.round(desiredDMDdiam_voxels*(projHeight_voxels/projWidth_voxels)))
    
    Dmax_mm = DMD_Ncols*pxSizeDMD / ( np.sin(degTilt * np.pi/180) + (projHeight_voxels/projWidth_voxels)*np.cos(degTilt * np.pi/180) )
    Hmax_mm = DMD_Nrows*pxSizeDMD / ( (projWidth_voxels/projHeight_voxels)*np.cos(degTilt * np.pi/180) + np.sin(degTilt * np.pi/180) )

    Dmax_vox = Dmax_mm/pxSizeDMD
    Hmax_vox = Hmax_mm/pxSizeDMD
    

    
    # # Given the rotation, what's the widest the image should be before rotation?
    # Dmax_vox = DMD_Nrows / ( np.sin(degTilt * np.pi/180) + (fT.shape[0]/fT.shape[1])*np.cos(degTilt * np.pi/180) )
    # Hmax_vox = DMD_Nrows / ( (fT.shape[1]/fT.shape[0])*np.cos(degTilt * np.pi/180) + np.sin(degTilt * np.pi/180) )
    
    # Dmax_mm  = Dmax_vox*pxSizeDMD*opticalMag
    # Hmax_mm  = Hmax_vox*pxSizeDMD*opticalMag
    
    
    if desiredDMDDiam_mm > Dmax_vox:
        print('Error! Image width, corrosponding to requested print size, does not fit onto DMD width after rotation.')
        print(' ')
        print('Maximum Possible Print Diameter = ' + str(Dmax_mm) + ' mm')
        print('Maximum Possible Print Height = '   + str(Hmax_mm) + ' mm')
        print(' ')
        
        print('A 1600x1600 DMD is assumed, because of square illumination')
        sys.exit()
    
    if desiredDMDheight_voxels > Hmax_vox:
        print('Error! Image height, corrosponding to requested print size, does not fit onto DMD height after rotation.')
        print(' ')
        print('Maximum Possible Print Diameter = ' + str(Dmax_mm) + ' mm')
        print('Maximum Possible Print Height = '   + str(Hmax_mm) + ' mm')
        print(' ')
        print('A 1600x1600 DMD is assumed, because of square illumination')
        sys.exit()
    
    return desiredDMDdiam_voxels, desiredDMDheight_voxels




def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):  # Won't call from main script
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = np.float32(cv2.resize(image, dim, interpolation = inter))

    # return the resized image
    return resized

def interpOntoDMD(g, desiredDMDdiam_voxels, degTilt):
    # Returns g, resclaed to desired DMD pixel diameter. No rotation however
    # Since DMD images need to be rotated, this won't be used to make final DMD iamges. It's for testing
    
    NcolsDMD           = 2560
    NrowsDMD           = 1600 #Resolution of VAM DMD. Vialux V9001
    
    gResized = np.zeros([g.shape[0],NrowsDMD,NcolsDMD])
    for ii in range(g.shape[0]):      
        temp    = image_resize(g[ii,:,:], width = desiredDMDdiam_voxels)
        
        
        if ~temp.shape[0]%2 and ~temp.shape[1]%2:   #If the new iamge (temp) is even in both dims
            gResized[ii, int(NrowsDMD/2-temp.shape[0]/2):int(temp.shape[0]/2+NrowsDMD/2),        int(NcolsDMD/2-temp.shape[1]/2):int(temp.shape[1]/2+NcolsDMD/2) ] = temp
            
        elif temp.shape[0]%2 and temp.shape[1]%2:   #If the new iamge (temp) is odd in both dims
            gResized[ii, int(NrowsDMD/2-temp.shape[0]//2-1):int(temp.shape[0]//2+NrowsDMD//2),   int(NcolsDMD/2-temp.shape[1]//2-1):int(temp.shape[1]//2+NcolsDMD//2) ] = temp
            
        elif ~temp.shape[0]%2 and temp.shape[1]%2:  # axis0 of temp is even, but axis1 of temp is odd
            gResized[ii, int(NrowsDMD/2-temp.shape[0]/2):int(temp.shape[0]/2+NrowsDMD/2),        int(NcolsDMD/2-temp.shape[1]//2-1):int(temp.shape[1]//2+NcolsDMD//2) ] = temp
            
        else:                                        # axis0 of temp is odd, and axis1 of temp is even
            gResized[ii, int(NrowsDMD/2-temp.shape[0]//2-1):int(temp.shape[0]//2+NrowsDMD//2),   int(NcolsDMD/2-temp.shape[1]/2):int(temp.shape[1]/2+NcolsDMD/2) ] = temp
        
        
    # gResized_Rotated = copy.copy(gResized)
    gResized_Rotated = gResized
    print('Only returning rotated image set')
    if degTilt != 0:
        for ii in range(g.shape[0]):
            gResized_Rotated[ii,:,:] = ndimage.rotate(gResized_Rotated[ii,:,:], degTilt, reshape=False) #Reshape = false means image will be clipped instead of making final array bigger
        
    return gResized, gResized_Rotated

def saveImageSetWithoutRotation(g,path, CCWrotationInDegrees):
    # g = image set
    # CCWrotationInDegrees - only for file naming !!!!!!!!!!!! This fxn does not apply rotation
    for ii in range(g.shape[0]):
        im = Image.fromarray( ((2**16-1)* g[ii,:,:]).astype(np.uint32))
        n = str(ii)       
        im.save(path+'\\'+n.zfill(4)+'_'+str(CCWrotationInDegrees)+'degCCW.png')

    return






































#--------- Unused functions written during development ------
    
# def rotate_image(mat, angle):                                                  # Won't call from main script
#     """
#     Rotates an image (angle in degrees) and expands image to avoid cropping
#     """

#     height, width = mat.shape[:2] # image shape has 3 dimensions
#     image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

#     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

#     # rotation calculates the cos and sin, taking absolutes of those.
#     abs_cos = abs(rotation_mat[0,0]) 
#     abs_sin = abs(rotation_mat[0,1])

#     # find the new width and height bounds
#     bound_w = int(height * abs_sin + width * abs_cos)
#     bound_h = int(height * abs_cos + width * abs_sin)

#     # subtract old image center (bringing image back to origo) and adding the new image center coordinates
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]

#     # rotate image with the new bounds and translated rotation matrix
#     rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
#     return rotated_mat








# def PrWin(f,voidInds,gelInds,normEachSlice, normToMinGel):
#     ftemp = copy.copy(f)
    
#     if normEachSlice:               
#         for ii in range(ftemp.shape[0]):
#             m = np.max(ftemp[ii,:,:])
#             if m != 0:
#                 ftemp[ii,:,:] = ftemp[ii,:,:]/m
#     else:
#         ftemp = ftemp/np.max(ftemp)
    
    
#     if normToMinGel:
#         minGelPerSlice = []
#         for ii in range(ftemp.shape[0]):  # For each slice
#             _, gelIndsInSlice   = getSlice_gel_void_inds(ii, voidInds, gelInds)
#             minGelDose          = np.min(ftemp[gelIndsInSlice[0],gelIndsInSlice[1],gelIndsInSlice[2]])
#             minGelPerSlice      = np.append(minGelPerSlice, [minGelDose], axis=0)
            
#         for ii in range(ftemp.shape[0]):  # For each slice
#             ftemp[ii,:,:] = ftemp[ii,:,:]/np.min(minGelPerSlice)
            
            
#     pw = (  np.min( ftemp[gelInds[0],gelInds[1],gelInds[2]] ) - np.max ( ftemp[voidInds[0],voidInds[1],voidInds[2]]) )/np.max(ftemp)
#     return pw


# def getPWperSlice(f,voidInds,gelInds,LTT, normg=0):
#     # g       = lttpy.project(fm,LTT)   
#     # g[g<0]  = 0

#     # f       = lttpy.backProject(g,LTT) 
#     # f       = f/np.max(f)
    
#     voidTuples  = np.transpose(voidInds)
#     gelTuples   = np.transpose(gelInds)
    
    
#     Nslices      = f.shape[0]
#     slicePWs     = []
#     minGelDoses  = []
#     maxVoidDoses = []
#     maxGelDoses  = []
#     for ii in range(Nslices):
#         # f_slice         = f[ii,:,:]
#         voidIndsInSlice = np.transpose( voidTuples[voidTuples[:,0]==ii,:] )
#         gelIndsInSlice  = np.transpose( gelTuples[gelTuples[:,0]==ii,:] )
        
#         pw, minGelDose, maxVoidDose, maxGelDose  = processWindow(0,f,voidIndsInSlice,gelIndsInSlice)
        
#         slicePWs        = np.append(slicePWs, [pw], axis=0)
#         minGelDoses     = np.append(minGelDoses, [minGelDose], axis=0)
#         maxVoidDoses    = np.append(maxVoidDoses, [maxVoidDose], axis=0)
#         maxGelDoses     = np.append(maxGelDoses, [maxGelDose], axis=0)

#     return slicePWs, minGelDoses, maxVoidDoses, maxGelDoses


# def getPWperSlice_from_g(g,voidInds,gelInds,LTT, normg=0):

#     f       = lttpy.backProject(g,LTT) 

    
    
#     Nslices      = f.shape[0]
#     slicePWs     = []
#     minGelDoses  = []
#     maxVoidDoses = []
#     maxGelDoses  = []
#     for ii in range(Nslices):
#         # f_slice         = f[ii,:,:]
#         # voidIndsInSlice = np.transpose( voidTuples[voidTuples[:,0]==ii,:] )
#         # gelIndsInSlice  = np.transpose( gelTuples[gelTuples[:,0]==ii,:] )
        
#         voidIndsInSlice, gelIndsInSlice         = getSlice_gel_void_inds(ii, voidInds, gelInds)
        
#         pw, minGelDose, maxVoidDose, maxGelDose  = processWindow(0,f,voidIndsInSlice,gelIndsInSlice)
        
#         slicePWs        = np.append(slicePWs, [pw], axis=0)
#         minGelDoses     = np.append(minGelDoses, [minGelDose], axis=0)
#         maxVoidDoses    = np.append(maxVoidDoses, [maxVoidDose], axis=0)
#         maxGelDoses     = np.append(maxGelDoses, [maxGelDose], axis=0)

#     return slicePWs, minGelDoses, maxVoidDoses, maxGelDoses


# def scale_g_slices(fm, voidInds,gelInds, Dcl,Dch, LTT):
    
#     g           = lttpy.project(fm,LTT)
#     g[g<0]      = 0
#     f           = lttpy.backProject(g,LTT)
#     f           = f/np.max(f)

    
#     _, minGelDoses, _, maxGelDoses = getPWperSlice(fm,voidInds,gelInds,LTT)
    
#     gScaled     = copy.copy(g)
#     for ii in range(fm.shape[0]):                    # For each slice
        
#         if (np.max(f[ii,:,:]) != 0):                 # If the slice is nonzero
#             # A               = 1/np.max(f[ii,:,:]) # Normalize by maxium of corrosponding f-slice
#             # # B               = maxGelDoses[ii]        # Normalize by maxium of slice, relative to all f-slice doses
#             # B               = np.max(minGelDoses)/minGelDoses[ii]
#             B               = np.min(minGelDoses[ii])
#             gScaled[:,ii,:] = g[:,ii,:]/B          # This g will reprocuce f correcly, expect for the nAngles factor
#             # gScaled[:,ii,:] = g[:,ii,:]/maxGelDoses[ii]
            
#     gScaled = gScaled/np.max(gScaled)                # Normalize all projection images to 1 for export and dmd intensity computation
    
#     f               = lttpy.backProject(gScaled,LTT) # All slices of f have the correct relative magntudes. 
#     f               = f/np.max(f)
    
#     return gScaled, f



