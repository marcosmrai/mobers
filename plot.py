# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:15:33 2015

@author: thalita
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare
from pmf import mf, plotPareto
from evaluation import ENSEMBLE_ORDER
from dbread import fold_load
import os
from pickle import load, dump

def loadfile(fname):
    with open(fname, 'rb') as f:
        return load(f)
def dumpfile(obj,fname):
    with open(fname, 'wb') as f:
        dump(obj,f)

'''
GLOBAL PARAMETERS
'''
MODELSFOLDER="models/"
RESULTSFOLDER="results/"
TOPK = 5
NFOLDS = 5
DIMS = [5,15,25]
E_ID = ENSMEBLE_ORDER

# Colors for bar graphs
color=['0.25','0.5','0.75']
#%% Training time plot and table
table = []
for  d_idx, d in enumerate(DIMS):
    time=[]
    numsol = []
    plt.figure(num=2*d_idx, figsize=(5,5))
    for fold in range(NFOLDS):
        times = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                         '-top%d-times.out'%TOPK)
        hours = np.array(times)/3600.0
        print 'pmf sum', hours.sum()
        totaltime = loadfile(MODELSFOLDER+'u-100k-fold-d%d-' % d +str(fold)+'runtime.out')
        print 'total nise', totaltime/3600.0
        numsol += [len(times)]
        time += [totaltime/3600.0]
        plt.subplot(np.ceil(NFOLDS/2.0), 2, fold+1)
        plt.plot(hours, label='fold %d'%(fold+1))
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.ylabel('training time (h)', fontsize='x-small')
        plt.xlabel('NISE iterations', fontsize='x-small')
        plt.legend(loc='best', fontsize='small')
        if fold == 0:
            plt.figure(num=2*d_idx+1, figsize=(5,2))
            plt.plot(hours*60)
            plt.title('Fold %d, d=%d' % (fold+1, d), fontsize='small')
            plt.yticks(fontsize='small')
            plt.xticks(fontsize='small')
            plt.ylabel('convergence time (min)', fontsize='small')
            plt.xlabel('NISE iterations', fontsize='small')
            plt.savefig(RESULTSFOLDER+'time_fold%d_d%d.png'%(fold, d))
            plt.figure(num=1)
    plt.figure(num=2*d_idx, figsize=(5,5))
    plt.tight_layout()
    plt.savefig(RESULTSFOLDER+'time_all_folds_d%d.png' % d)
    table += [time, numsol]

np.savetxt(RESULTSFOLDER+'table_numsol_hours.txt', np.array(table).T,
           fmt='%0.1f', delimiter=' & ', newline='\\\\ \hline \n')

#%% Lambda and CV precisions table
table=[]
for d in DIMS:
    L = []
    P = []
    for fold in range(NFOLDS):
        result = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                          '-top%d-results.out'%TOPK)
        print len(result[0:-3])
        precisions = [r[2] for r in result[0:-3]]
        model_id = np.argmax(precisions)
        L.append(result[model_id][1])
        P.append(precisions[model_id])
    table += [L, P]
np.savetxt(RESULTSFOLDER+'table_lambda_precision.txt', np.array(table).T,
           fmt='%0.4f', delimiter=' & ', newline='\\\\ \hline \n')


#%% Precision bar plots for all folds together with friedman test. Prints precision information to be used in R
tables = []
for d in DIMS:
    pBetter=[]
    pVote=[]
    pWeigh=[]

    for fold in range(NFOLDS):
        out = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%TOPK)
        #ordem: maioria, ponderado, best
        #print out[-1]
        pBetter.append(out[-1][2])
        pWeigh.append(out[-2][1])
        pVote.append(out[-3][1])

    print 'Friedman', friedmanchisquare(np.array(pBetter),
                             np.array(pWeigh),
                             np.array(pVote))

    print 'Precisions d=%d'%d
    Rprint = lambda alist :  ' = c(' + \
                           ", ".join([str(it) for it in alist]) + ')'
    folds = [i for i in range(NFOLDS)]*3
    precisions = pBetter + pWeigh + pVote
    ids = ["'best'"]*NFOLDS + ["'weight'"]*NFOLDS + ["'vote'"]*NFOLDS
    print 'datafold', Rprint(folds)
    print 'precision', Rprint(precisions)
    print 'algo', Rprint(ids)

    tables.append(np.vstack((np.array(pBetter),
                             np.array(pWeigh),
                             np.array(pVote) )).T)
    x=np.arange(1,NFOLDS+1,1)

    y = [4, 9, 2,5,6]
    z=[1,2,3,5,7]
    k=[11,12,13,5,9]
    plt.figure()
    ax = plt.subplot(111)
    w = 0.3
    ax.bar(x-w, pVote,width=w,color=color[0],align='center', label='Vote')
    ax.bar(x, pWeigh,width=w,color=color[1],align='center', label='Weighted')
    ax.bar(x+w, pBetter,width=w,color=color[2],align='center', label='Best')
    ax.autoscale(tight=True)
    ax.legend(loc=4)
    plt.ylim((0.7,1.0))
    #plt.ylim((0,0.9))
    plt.xticks(range(1,NFOLDS+1),['Fold %d'%i for i in range(1,NFOLDS+1)])
    plt.title('Precision (d=%d)'%d)
    plt.savefig(RESULTSFOLDER+'precisionat%d_bars_d%d.png'%(TOPK, d))

table = np.hstack((tables[0],tables[1]))
table = np.vstack((table,table.mean(axis=0), table.std(axis=0)))

np.savetxt(RESULTSFOLDER+'table_precisions.txt', table, fmt='%.4f', delimiter=' & ', newline='\\\\ \hline \n')


#%% Average precision bar plots with error bars
plt.figure()
for d_i, d in enumerate(DIMS):
    best = []
    vote = []
    weight = []
    for fold in range(NFOLDS):
        out = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%TOPK)
        best += [out[-1][2]]
        vote += [out[-3][1]]
        weight += [out[-2][1]]


    data = np.array([vote,weight,best])
    labels = ['Vote','Weighted','Best']
    colors = color
    plt.subplot(1,3,d_i+1)
    plt.title('Precision (d=%d)'%d)
    for i in range(3):
        plt.errorbar(i,data[i,:].mean(),data[i,:].std(),linewidth=2, color='k')
        plt.bar(i-0.4, data[i,:].mean(), label=labels[i], color=colors[i])
        plt.ylim((0.8,0.95))
    plt.xticks(range(3),labels)
plt.savefig(RESULTSFOLDER+'precisionat%d_errorbars.png'%(TOPK))


#%% Pareto plot
for d in DIMS:
    fold=0
    out = loadfile(MODELSFOLDER+'u-100k-fold-d%d-%d.out'%(d, fold))
    plt.figure()
    plotPareto(out)
    plt.title('NISE solutions (d=%d)'%d)

    result = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%TOPK)
    precisions = [r[2] for r in result[0:-3]]
    model_id = np.argmax(precisions)
    alambda = result[model_id][1]

    errs = out[model_id].objs
    xticks = plt.gca().get_xticks()
    dx = xticks[1] - xticks[0]
    #plt.text(errs[0]+dx, errs[1],)
    #plt.arrow(errs[0]+dx, errs[1],-dx,0, head_width=1)
    plt.annotate('$\lambda=%0.4f$ (best in CV)'%alambda,
                xy=(errs[0], errs[1]), xytext=(errs[0]+dx, errs[1]),
                arrowprops=dict(facecolor='gray', shrink=0.025))
    plt.savefig(RESULTSFOLDER+'pareto_fold%d_train_d%d'%(fold, d))


