# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:24:03 2020

@author: chra3427

Module for setting up LTT
"""

import numpy as np
import os
# os.chdir('C:/LTT/LTT_Windows_v1.6.12/python') #Add LTT python dll's to path
#os.chdir('C:/LTT/LTT_v1.6.25/python') #Add LTT python dll's to path
os.chdir('C:/Users/Admin.DESKCOMPUTER/Desktop/LTT_v1.6.30/python') #Add LTT python dll's to path
from LTTserver import LTTserver



def setup_LTT(LTT,angularRange,nAngles,nrays,pxsize,alpha):
    
    LTT.cmd('archdir = pwd')
    LTT.cmd('dataType = atten')
    LTT.cmd('geometry = parallel')
    LTT.cmd('arange = ' + str(angularRange))
    LTT.cmd('nangles = ' + str(nAngles))
    LTT.cmd('pxsize = ' + str(pxsize))
    LTT.cmd('pzsize = 0.2')
    LTT.cmd('nrays = ' + str(nrays))
    LTT.cmd('nslices = 1')
    LTT.cmd('pxcenter = (numCols-1)/2')
    LTT.cmd('pzcenter = (numRows-1)/2')
    LTT.cmd('defaultVolume')
    LTT.cmd('diskIO = off')
    LTT.cmd('exponentialRadonCoeff = ' + str(alpha)); 
    LTT.cmd('whichProjector = SF');
    
    return LTT



def setup_LTT3D(LTT,angularRange,nAngles,nrays,alpha,pxsize, nslices):
    LTT.cmd('archdir = pwd')
    LTT.cmd('dataType = atten')
    LTT.cmd('geometry = parallel')
    LTT.cmd('arange = ' + str(angularRange))
    LTT.cmd('nangles = ' + str(nAngles))
    LTT.cmd('pxsize = ' + str(pxsize))
    LTT.cmd('pzsize = 0.2')
    LTT.cmd('nrays = ' + str(nrays))
    LTT.cmd('nslices = '+str(nslices))
    LTT.cmd('pxcenter = (numCols-1)/2')
    LTT.cmd('pzcenter = (numRows-1)/2')
    LTT.cmd('defaultVolume')
    LTT.cmd('diskIO = off')
    LTT.cmd('exponentialRadonCoeff = ' + str(alpha)); 
    LTT.cmd('whichProjector = SF');
    
    return LTT





def project(volData,LTT=None):  #Good ol project fxn
    if LTT is None:
        LTT = LTTserver()
        
    LTT.setAllReconSlicesZ(np.ascontiguousarray(volData, dtype=np.float32))
    LTT.cmd('project')
    projections = LTT.getAllProjections()
    
    return projections



def backProject(projData,LTT=None):
    if LTT is None:
        LTT = LTTserver()
        
    LTT.setAllProjections(projData)
    LTT.cmd('backproject')
    reconstruction = LTT.getAllReconSlicesZ()

    return reconstruction


def FBP(projData, LTT=None):
    if LTT is None:
        LTT = LTTserver()
        
    LTT.setAllProjections(projData)
    LTT.cmd('FBP')
    reconstruction = LTT.getAllReconSlicesZ()

    return reconstruction


def clippedFBP(projData, LTT=None):
    if LTT is None:
        LTT = LTTserver()
        
        
    LTT.setAllProjections(projData)
    LTT.cmd('FBP {clip=true}')
    reconstruction = LTT.getAllReconSlicesZ()

    return reconstruction



def runCAL(fT,g0,N_iter,sigmoid,rotationRate,Dc,resinDiameter,beta,calcDoseMap,LTT=None):
    if LTT is None:
        LTT = LTTserver()
        
    LTT.setAllReconSlicesZ(fT);          #put target, f, into volume data slot
    LTT.setAllProjections(g0);       #put g0, our initial guess, into projection slot
    
    if calcDoseMap == 1:   
         dmstr = 'true'     # Calc a dosemap, including scaping based on alpha, omega, etc
    else:
         dmstr = 'false'    # Simply return the backprojection of the last interation
    
    LTT.cmd('CAL {N_iter=' + str(N_iter) +'; sigmoidType=poly; sigmoidParameter=' + str(sigmoid) + 
            '; rotationRate=' + str(rotationRate) + '; criticalDose=' + str(Dc) +
            '; resinDiameter=' + str(resinDiameter) + '; beta=' + str(beta) +
            '; calculateDoseMap=' +dmstr+'; useDegPerSec=false; seedWithmemory=true}')
   
    Pstar_g  = LTT.getAllReconSlicesZ();
    return Pstar_g




