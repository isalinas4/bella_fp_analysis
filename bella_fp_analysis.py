# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:19:12 2021

@author: isabe
"""

import numpy as np
from sklearn.metrics import auc
import matplotlib  # standard Python plotting library
import scipy.stats as stats
import tdt
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['font.size'] = 16 #set font size for all plots

#Need to change to be able to loop through 


def zscore(mouse,dates): 

    for d in dates:
        REF_EPOC = 'Mi1/' #event store name. This holds behavioral codes that are 
        # read through ports A & B on the front of the RZ
        SHOCK_CODE = [1] #shock onset event code we are interested in
        
        # make some variables up here to so if they change in new recordings you won't
        # have to change everything downstream
        ISOS = '_405G' # 405nm channel. Formally STREAM_STORE1 in maltab example
        GCaMP = '_490G' # 465nm channel. Formally STREAM_STORE2 in maltab example
        TRANGE = [-600, 1900] # window size [start time relative to epoc onset, window duration]
        BASELINE_PER = [-500, -10] # baseline period within our window
        ARTIFACT = float("inf") # optionally set an artifact rejection level
        
        #call read block - new variable 'data' is the full data structure
        #Indentify Mice
        expt = 'Drug'
        cohort = 'GAD2 Only'
        exptTitle ='FP in ' + cohort + '. Check Signal ' + d
        BLOCKPATH = os.path.join('C:\\Users\\isabe\\Dropbox (University of Oregon)\\UO-Sylwestrak Lab\TDT\Tanks', expt + '-' + d , mouse)
        data = tdt.read_block(BLOCKPATH)
        
        data = tdt.epoc_filter(data, REF_EPOC, t=TRANGE, values=SHOCK_CODE)
        
        # More examples of list comprehensions
        min1 = np.min([np.size(x) for x in data.streams[GCaMP].filtered])
        min2 = np.min([np.size(x) for x in data.streams[ISOS].filtered])
        data.streams[GCaMP].filtered = [x[1:min1] for x in data.streams[GCaMP].filtered]
        data.streams[ISOS].filtered = [x[1:min2] for x in data.streams[ISOS].filtered]
        
        # Downsample and average 10x via a moving window mean
        N = 20 # Average every 10 samples into 1 value
        F405 = []
        F465 = []
        for lst in data.streams[ISOS].filtered: 
            small_lst = []
            for i in range(0, min2, N):
                small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
            F405.append(small_lst)
        
        for lst in data.streams[GCaMP].filtered: 
            small_lst = []
            for i in range(0, min1, N):
                small_lst.append(np.mean(lst[i:i+N-1]))
            F465.append(small_lst)
            
        #Create a mean signal, standard error of signal, and DC offset
        meanF405 = np.mean(F405, axis=0)
        stdF405 = np.std(F405, axis=0)/np.sqrt(len(data.streams[ISOS].filtered))
        dcF405 = np.mean(meanF405)
        meanF465 = np.mean(F465, axis=0)
        stdF465 = np.std(F465, axis=0)/np.sqrt(len(data.streams[GCaMP].filtered))
        dcF465 = np.mean(meanF465)
        
        # Create the time vector for each stream store
        ts1 = TRANGE[0] + np.linspace(1, len(meanF465), len(meanF465))/data.streams[GCaMP].fs*N
        ts2 = TRANGE[0] + np.linspace(1, len(meanF405), len(meanF405))/data.streams[ISOS].fs*N
        
        meanF405 = meanF405 - dcF405
        meanF465 = meanF465 - dcF465
        
        
        fig = plt.figure(figsize=(9, 14))
        
        Y_fit_all = []
        Y_dF_all = []
        for x, y in zip(F405, F465):
            x = np.array(x)
            y = np.array(y)
            bls = np.polyfit(x, y, 1)
            fit_line = np.multiply(bls[0], x) + bls[1]
            Y_fit_all.append(fit_line)
            Y_dF_all.append(y-fit_line)
        
        # Getting the z-score and standard error
        zall = []
        for dF in Y_dF_all: 
            ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
            zb = np.mean(dF[ind])
            zsd = np.std(dF[ind])
            zall.append((dF - zb)/zsd)
        
        zerror = np.std(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
        
        
        plt.figure(3, figsize=(15, 4), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 2, wspace=.3)
        fig.suptitle(mouse)
        
        plt.plot(ts2, np.mean(zall, axis=0), linewidth=2, label= d)
        plt.fill_between(ts1, np.mean(zall, axis=0)+zerror
                              ,np.mean(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
        plt.axvline(x=0, linewidth=3, color='red', linestyle='--')
        plt.ylabel('z-Score')
        plt.xlabel('Time (s) after IP injection')
        plt.xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
        plt.title( mouse + ' ' +  d + ' Nlx/Sal Response')
        plt.legend( )
        #plt.savefig('Users\isabe\Documents\Emily\ '+  mouse +'_'+ d +'.png', dpi = 300)
    plt.show()


    
def AUC(mouse, date):
    
    REF_EPOC = 'Mi1/' #event store name. This holds behavioral codes that are 
    # read through ports A & B on the front of the RZ
    SHOCK_CODE = [1] #shock onset event code we are interested in
    
    # make some variables up here to so if they change in new recordings you won't
    # have to change everything downstream
    ISOS = '_405G' # 405nm channel. Formally STREAM_STORE1 in maltab example
    GCaMP = '_490G' # 465nm channel. Formally STREAM_STORE2 in maltab example
    TRANGE = [-600, 1900] # window size [start time relative to epoc onset, window duration]
    BASELINE_PER = [-500, -10] # baseline period within our window
    ARTIFACT = float("inf") # optionally set an artifact rejection level
    
    #call read block - new variable 'data' is the full data structure
    #Indentify Mice
    expt = 'Drug'
    cohort = 'GAD2 Only'
    exptTitle ='FP in ' + cohort + '. Check Signal ' + date
    BLOCKPATH = os.path.join('C:\\Users\\isabe\\Dropbox (University of Oregon)\\UO-Sylwestrak Lab\TDT\Tanks', expt + '-' + date , mouse)
    data = tdt.read_block(BLOCKPATH)
    
    data = tdt.epoc_filter(data, REF_EPOC, t=TRANGE, values=SHOCK_CODE)
    
    # More examples of list comprehensions
    min1 = np.min([np.size(x) for x in data.streams[GCaMP].filtered])
    min2 = np.min([np.size(x) for x in data.streams[ISOS].filtered])
    data.streams[GCaMP].filtered = [x[1:min1] for x in data.streams[GCaMP].filtered]
    data.streams[ISOS].filtered = [x[1:min2] for x in data.streams[ISOS].filtered]
    
    # Downsample and average 10x via a moving window mean
    N = 20 # Average every 10 samples into 1 value
    F405 = []
    F465 = []
    for lst in data.streams[ISOS].filtered: 
        small_lst = []
        for i in range(0, min2, N):
            small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
        F405.append(small_lst)
    
    for lst in data.streams[GCaMP].filtered: 
        small_lst = []
        for i in range(0, min1, N):
            small_lst.append(np.mean(lst[i:i+N-1]))
        F465.append(small_lst)
        
    #Create a mean signal, standard error of signal, and DC offset
    meanF405 = np.mean(F405, axis=0)
    stdF405 = np.std(F405, axis=0)/np.sqrt(len(data.streams[ISOS].filtered))
    dcF405 = np.mean(meanF405)
    meanF465 = np.mean(F465, axis=0)
    stdF465 = np.std(F465, axis=0)/np.sqrt(len(data.streams[GCaMP].filtered))
    dcF465 = np.mean(meanF465)
    
    # Create the time vector for each stream store
    ts1 = TRANGE[0] + np.linspace(1, len(meanF465), len(meanF465))/data.streams[GCaMP].fs*N
    ts2 = TRANGE[0] + np.linspace(1, len(meanF405), len(meanF405))/data.streams[ISOS].fs*N
    
    meanF405 = meanF405 - dcF405
    meanF465 = meanF465 - dcF465
    
    
    fig = plt.figure(figsize=(9, 14))
    
    Y_fit_all = []
    Y_dF_all = []
    for x, y in zip(F405, F465):
        x = np.array(x)
        y = np.array(y)
        bls = np.polyfit(x, y, 1)
        fit_line = np.multiply(bls[0], x) + bls[1]
        Y_fit_all.append(fit_line)
        Y_dF_all.append(y-fit_line)
    
    # Getting the z-score and standard error
    zall = []
    for dF in Y_dF_all: 
        ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
        zb = np.mean(dF[ind])
        zsd = np.std(dF[ind])
        zall.append((dF - zb)/zsd)
    
    zerror = np.std(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    AUC = [] # cue, shock
    ind1 = np.where((np.array(ts2)<-3) & (np.array(ts2)>-5))
    AUC1= auc(ts2[ind1], np.mean(zall, axis=0)[ind1])
    ind2 = np.where((np.array(ts2)>0) & (np.array(ts2)<2))
    AUC2= auc(ts2[ind2], np.mean(zall, axis=0)[ind2])
    AUC.append(AUC1)
    AUC.append(AUC2)
    
    # Run a two-sample T-test
    t_stat,p_val = stats.ttest_ind(np.mean(zall, axis=0)[ind1],
                                    np.mean(zall, axis=0)[ind2], equal_var=False)
    ax3 = fig.add_subplot(414)
    p9 = ax3.bar(np.arange(len(AUC)), AUC, color=[.8, .8, .8], align='center', alpha=0.5)
    
    # statistical annotation
    x1, x2 = 0, 1 # columns indices for labels
    y, h, col = max(AUC) + 2, 2, 'k'
    ax3.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    p10 = ax3.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    
    # Finish up the plot
    ax3.set_ylim(0,y+2*h)
    ax3.set_ylabel('AUC')
    ax3.set_title(mouse + ' Response to Nlx/ Sal')
    ax3.set_xticks(np.arange(-1, len(AUC)+1))
    ax3.set_xticklabels(['','before IP','After IP',''])
    
    fig.tight_layout()
    fig





# # Start making a figure with 4 subplots
# # First plot is the 405 and 465 averaged signals
# # ax0 = fig.add_subplot(411) # work with axes and not current plot (plt.)

# # # Plotting the traces
# # p1, = ax0.plot(ts1, meanF465, linewidth=2, color='green', label='GCaMP')
# # p2, = ax0.plot(ts2, meanF405, linewidth=2, color='blueviolet', label='ISOS')

# # # Plotting standard error bands
# # p3 = ax0.fill_between(ts1, meanF465+stdF465, meanF465-stdF465,
# #                       facecolor='green', alpha=0.2)
# # p4 = ax0.fill_between(ts2, meanF405+stdF405, meanF405-stdF405,
# #                       facecolor='blueviolet', alpha=0.2)

# # # Plotting a line at t = 0
# # p5 = ax0.axvline(x=0, linewidth=3, color='slategray', label='Shock Onset')

# # # Finish up the plot
# # ax0.set_xlabel('Seconds')
# # ax0.set_ylabel('mV')
# # ax0.set_title('Foot Shock Response, %i Trials (%i Artifacts Removed)'
# #               % (len(data.streams[GCaMP].filtered), total_artifacts))
# # ax0.legend(handles=[p1, p2, p5], loc='upper right')
# # ax0.set_ylim(min(np.min(meanF465-stdF465), np.min(meanF405-stdF405)),
# #              max(np.max(meanF465+stdF465), np.max(meanF405+stdF405)))
# # ax0.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0]);

# # plt.show() 


