# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:15:33 2015

@author: thalita
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pickle import load, dump


def loadfile(fname):
    with open(fname, 'rb') as f:
        return load(f)
def dumpfile(obj,fname):
    with open(fname, 'wb') as f:
        dump(obj,f)

from pmf import ProbabilisticMatrixFactorization, plotPareto
from dbread import fold_load
import os

from scipy.stats import friedmanchisquare

folder="results_31_08_2015/"
topk = 5
color=['0.25','0.5','0.75']
nfolds = 5
DIMS = [5,15]
#%%
table = []
for  d in DIMS:
    time=[]
    numsol = []
    plt.figure(num=d/50, figsize=(5,5))
    for fold in range(nfolds):
        times = loadfile(folder+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                         '-top%d-times.out'%topk)
        hours = np.array(times)/3600.0
        print 'pmf sum', hours.sum()
        totaltime = loadfile('models/u-100k-fold-d%d-' % d +str(fold)+'runtime.out')
        print 'total nise', totaltime/3600.0
        numsol += [len(times)]
        time += [totaltime/3600.0]
        '''
        plt.subplot(3, 2, fold+1)
        plt.plot(hours, label='fold %d'%(fold+1))
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.ylabel('training time (h)', fontsize='x-small')
        plt.xlabel('NISE iterations', fontsize='x-small')
        plt.legend(loc='best', fontsize='small')
        '''
        if fold == 0:
            plt.figure(num=d/50+1, figsize=(5,2))
            plt.plot(hours*60)
            plt.title('Fold %d, d=%d' % (fold+1, d), fontsize='small')
            plt.yticks(fontsize='small')
            plt.xticks(fontsize='small')
            plt.ylabel('convergence time (min)', fontsize='small')
            plt.xlabel('NISE iterations', fontsize='small')
            plt.savefig(folder+'time_fold%d_d%d.png'%(fold, d))
            plt.figure(num=1)
            #    plt.tight_layout()
    table += [time, numsol]

np.savetxt(folder+'table_numsol_hours.txt', np.array(table).T,
           fmt='%0.1f', delimiter=' & ', newline='\\\\ \hline \n')

#%%
table=[]
for d in DIMS:
    L = []
    P = []
    for fold in range(nfolds):
        result = loadfile(folder+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                          '-top%d-results.out'%topk)
        print len(result[0:-3])
        precisions = [r[2] for r in result[0:-3]]
        model_id = np.argmax(precisions)
        L.append(result[model_id][1])
        P.append(precisions[model_id])
    table += [L, P]
np.savetxt(folder+'table_lambda_precision.txt', np.array(table).T,
           fmt='%0.4f', delimiter=' & ', newline='\\\\ \hline \n')

#%%

for d in DIMS:
    fold=0
    train,trainU,trainI,valid,validU,validI,test,testU,testI=\
        fold_load('ml-100k',fold)
    out = loadfile('models/u-100k-fold-d%d-%d.out'%(d, fold))
    plt.figure()
    plotPareto(out,train)
    plt.title('NISE solutions (d=%d)'%d)


    result = loadfile(folder+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%topk)
    precisions = [r[2] for r in result[0:-3]]
    model_id = np.argmax(precisions)
    alambda = result[model_id][1]

    errs = out[model_id].objErrors(train)
    plt.text(errs[0]*1.01, errs[1],'$\lambda=%0.4f$ (best in CV)'%alambda)
    plt.arrow(errs[0]*1.01, errs[1],-0.01*errs[0],0)
    plt.savefig(folder+'pareto_fold%d_train_d%d'%(fold, d))



#%%
tables = []
for d in DIMS:
    pBetter=[]
    pVote=[]
    pWeigh=[]

    for fold in range(nfolds):
        with open(folder+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%topk, 'rb') as handle:
            out = pickle.load(handle)
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
    folds = [i for i in range(nfolds)]*3
    precisions = pBetter + pWeigh + pVote
    ids = ["'best'"]*nfolds + ["'weight'"]*nfolds + ["'vote'"]*nfolds
    print 'datafold', Rprint(folds)
    print 'precision', Rprint(precisions)
    print 'algo', Rprint(ids)

    tables.append(np.vstack((np.array(pBetter),
                             np.array(pWeigh),
                             np.array(pVote) )).T)
    x=np.arange(1,nfolds+1,1)

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
    plt.xticks(range(1,nfolds+1),['Fold %d'%i for i in range(1,nfolds+1)])
    plt.title('Precision (d=%d)'%d)
    plt.savefig(folder+'precisionat%d_bars_d%d.png'%(topk, d))

table = np.hstack((tables[0],tables[1]))
table = np.vstack((table,table.mean(axis=0), table.std(axis=0)))

np.savetxt(folder+'table_precisions.txt', table, fmt='%.4f', delimiter=' & ', newline='\\\\ \hline \n')


#%%
plt.figure()
for d_i, d in enumerate(DIMS):
    best = []
    vote = []
    weight = []
    for fold in range(nfolds):
        out = loadfile(folder+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%topk)
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
plt.savefig(folder+'precisionat%d_errorbars.png'%(topk))

