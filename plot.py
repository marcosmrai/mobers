# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:15:33 2015

@author: thalita
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare
from pmf import mf
from evaluation import ENSEMBLE_ORDER
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
E_ID = ENSEMBLE_ORDER

# Colors for bar graphs
COLORS=[str(x) for x in np.logspace(-0.8,-0.15,len(E_ID))]


def time_plots():
    '''
    Training time plot and table
    '''
    table = []
    for  d_idx, d in enumerate(DIMS):
        time=[]
        numsol = []
        plt.figure(num=2*d_idx, figsize=(5,5))
        for fold in range(NFOLDS):
            times = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                             '-top%d-times.out'%TOPK)
            minutes = np.array(times)/60.0
            totaltime = loadfile(MODELSFOLDER+'u-100k-fold-d%d-' % d +str(fold)+'runtime.out')
            numsol += [len(times)]
            time += [totaltime/60.0]
            plt.subplot(np.ceil(NFOLDS/2.0), 2, fold+1)
            plt.plot(minutes, label='fold %d'%(fold+1))
            plt.yticks(fontsize='small')
            plt.xticks(fontsize='small')
            plt.ylabel('training time (min)', fontsize='x-small')
            plt.xlabel('NISE iterations', fontsize='x-small')
            plt.legend(loc='best', fontsize='small')
            if fold == 0:
                plt.figure(num=2*d_idx+1, figsize=(5,3))
                plt.plot(minutes)
                plt.title('Fold %d, d=%d' % (fold+1, d), fontsize='small')
                plt.yticks(fontsize='small')
                plt.xticks(fontsize='small')
                plt.ylabel('convergence time (min)', fontsize='small')
                plt.xlabel('NISE iterations', fontsize='small')
                plt.savefig(RESULTSFOLDER+'time_fold%d_d%d.png'%(fold, d))
                plt.figure(num=2*d_idx, figsize=(5,5))
        plt.tight_layout()
        plt.savefig(RESULTSFOLDER+'time_all_folds_d%d.png' % d)
        table += [time, numsol]

    np.savetxt(RESULTSFOLDER+'table_numsol_minutes.txt', np.array(table).T,
               fmt='%0.1f', delimiter=' & ', newline='\\\\ \hline \n')

def lambda_precisions_table():
    '''
    Lambda and CV precisions table
    '''
    table=[]
    for d in DIMS:
        L = []
        P = []
        for fold in range(NFOLDS):
            result = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+\
                              '-top%d-results.out'%TOPK)
            precisions = [r[2] for r in result[0:-len(E_ID)]]
            model_id = np.argmax(precisions)
            L.append(result[model_id][1])
            P.append(precisions[model_id])
        table += [L, P]
    np.savetxt(RESULTSFOLDER+'table_lambda_precision.txt', np.array(table).T,
               fmt='%0.4f', delimiter=' & ', newline='\\\\ \hline \n')

def precision_plot_table():
    '''
    Precision bar plots for all folds together with friedman test.
    Prints precision information to be used in R
    '''
    tables = []
    for d in DIMS:
        pBetter=[]
        pVote=[]
        pWeigh=[]
        pFvote=[]

        for fold in range(NFOLDS):
            out = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%TOPK)
            #ordem: maioria, ponderado, best
            #print out[-1]
            pBetter.append(out[E_ID['best']][2])
            pWeigh.append(out[E_ID['weighted']][1])
            pVote.append(out[E_ID['vote']][1])
            pFvote.append(out[E_ID['filtered']][1])

        print 'Friedman', friedmanchisquare(np.array(pBetter),
                                 np.array(pWeigh),
                                 np.array(pVote),
                                 np.array(pFvote))

        print 'Precisions d=%d'%d
        Rprint = lambda alist :  ' = c(' + \
                               ", ".join([str(it) for it in alist]) + ')'
        folds = [i for i in range(NFOLDS)]*3
        precisions = pBetter + pWeigh + pVote + pFvote
        ids = ["'best'"]*NFOLDS + ["'weight'"]*NFOLDS + \
              ["'vote'"]*NFOLDS + ["'filtered'"]*NFOLDS
        print 'datafold', Rprint(folds)
        print 'precision', Rprint(precisions)
        print 'algo', Rprint(ids)

        tables.append(np.vstack((np.array(pBetter),
                                 np.array(pWeigh),
                                 np.array(pVote),
                                 np.array(pFvote) )).T)
        x=np.arange(1,NFOLDS+1,1)

        y = [4, 9, 2,5,6]
        z=[1,2,3,5,7]
        k=[11,12,13,5,9]
        plt.figure()
        ax = plt.subplot(111)
        w = 0.2
        ax.bar(x-2*w, pVote,width=w,color=COLORS[0],align='center', label='Vote')
        ax.bar(x-w, pFvote,width=w,color=COLORS[1],align='center', label='Filtered')
        ax.bar(x, pWeigh,width=w,color=COLORS[2],align='center', label='Weighted')
        ax.bar(x+w, pBetter,width=w,color=COLORS[3],align='center', label='Best')
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


def avg_precision_plot():
    '''
    Average precision bar plots with error bars
    '''
    plt.figure(figsize=(10,6))
    for d_i, d in enumerate(DIMS):
        best = []
        vote = []
        fvote = []
        weight = []
        for fold in range(NFOLDS):
            out = loadfile(RESULTSFOLDER+'u-100k-fold-'+str(d)+'-d'+str(fold)+'-top%d-results.out'%TOPK)
            best += [out[E_ID['best']][2]]
            vote += [out[E_ID['vote']][1]]
            fvote += [out[E_ID['filtered']][1]]
            weight += [out[E_ID['weighted']][1]]


        data = np.array([vote,fvote,weight,best])
        labels = ['Vote','Filtered','Weighted','Best']
        plt.subplot(1,len(DIMS),d_i+1)
        plt.title('Precision (d=%d)'%d)
        w = 0.8
        for i in range(data.shape[0]):
            plt.errorbar(i,data[i,:].mean(),data[i,:].std(),linewidth=2, color='k')
            plt.bar(i-w*0.5, data[i,:].mean(), width=w,
                    label=labels[i], color=COLORS[i])
            plt.ylim((0.8,0.95))
        plt.xticks(range(data.shape[0]),labels, fontsize='x-small')
    plt.savefig(RESULTSFOLDER+'precisionat%d_errorbars.png'%(TOPK))


def plotPareto(list_,figpath=None):
    plotL=np.array([i.objs for i in list_])
    plt.plot(plotL[:,0],plotL[:,1],'ok')
    plt.xlabel('MSE objective')
    plt.ylabel('Regularization objective (Avg. L2 norm)')
    if figpath != None:
        plt.savefig(figpath)

def _single_annotated_pareto(d):
    '''
    Pareto plot with best lambda annotated
    '''
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
    plt.annotate('$\lambda=%0.4f$ (best in CV)'%alambda,
                xy=(errs[0], errs[1]), xytext=(errs[0]+dx, errs[1]),
                arrowprops=dict(facecolor='gray', shrink=0.025))
    plt.savefig(RESULTSFOLDER+'pareto_fold%d_train_d%d'%(fold, d))

def annotated_pareto(d=None):
    '''
    Do single or all pareto plots
    '''
    if d != None:
        _single_annotated_pareto(d)
    else:
        map(_single_annotated_pareto, DIMS)

if __name__=='__main__':
    time_plots()
    lambda_precisions_table()
    precision_plot_table()
    avg_precision_plot()
    annotated_pareto()

