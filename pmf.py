import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy
from numpy.random import shuffle
import pickle
import copy
import bisect

from time import time

from dbread import *
import multiprocessing

import scipy.optimize as opt

class mf():
    def __init__(self, rTrain, nUsers, nItems, d=50, lambdaa=0.1,listt = None): 
        self.d = d
        self.lambdaa = lambdaa

        self.rTrain = rTrain

        self.nU = nUsers
        self.nI = nItems
        self.nR = len(rTrain)

        self.training_time = 0
        self.theta = numpy.ones(self.nU*self.d + self.nI*self.d)/numpy.sqrt(self.d)

        self.objs = [self.errorObj(self.theta),self.regObj(self.theta)]
        self.obj = (1-self.lambdaa)*self.objs[0]+(self.lambdaa)*self.objs[1]

        if listt != None:
            newObj = self.obj
            nindex = -1
            for index,model in enumerate(listt):
                obj = (1-self.lambdaa)*model.objs[0]+(self.lambdaa)*model.objs[1]
                if obj < newObj:
                    nindex = index
                    newObj = obj
            if nindex >=0:
                self.theta = copy.deepcopy(listt[nindex].theta)
                self.obj = copy.deepcopy(listt[nindex].obj)
                self.objs = copy.deepcopy(listt[nindex].objs)

    # return the regularization parameter
    def getReg(self):
        return self.lambdaa


    # calculates the error for a rating
    def sampleError(self, rating, theta):
        u,i,score = rating
        uIndS = self.d*u;uIndF = self.d*(u+1);
        iIndS = self.nU*self.d + self.d*i; iIndF = self.nU*self.d + self.d*(i+1); 
        uv = theta[uIndS:uIndF]
        iv = theta[iIndS:iIndF]
        return numpy.dot(uv,iv)-score
        
    # calculates the error objective
    def errorObj(self,theta):
        obj=0
        for rating in self.rTrain:
            obj+=self.sampleError(rating,theta)**2

        return obj/self.nR

    # calculates the regularization objective
    def regObj(self,theta):
        return np.dot(theta,theta)/theta.size

    # calculates the scalarized objective 
    def fObj(self,theta):
        return (1-self.lambdaa)*self.errorObj(theta)+self.lambdaa*self.regObj(theta)

    # calculates the gradient for the error objective
    def errorGrad(self,theta):
        grad = numpy.zeros(theta.shape)

        for rating in self.rTrain:
            u,i,score = rating
            uIndS = self.d*u;uIndF = self.d*(u+1);
            iIndS = self.nU*self.d + self.d*i; iIndF = self.nU*self.d + self.d*(i+1); 
            error = self.sampleError(rating,theta)
            grad[uIndS:uIndF] += 2*theta[iIndS:iIndF]*error
            grad[iIndS:iIndF] += 2*theta[uIndS:uIndF]*error

        return grad/self.nR

    # calculates the gradient for the regularization objective
    def regGrad(self,theta):
        grad = 2*theta
        return grad/theta.size

    # calculates the gradient for the scalarized objective
    def fGrad(self,theta):
        return (1-self.lambdaa)*self.errorGrad(theta)+self.lambdaa*self.regGrad(theta)

    # calculates the gradient for the error objective
    def errorHessD(self,theta,d):
        hessD = numpy.zeros(theta.shape)

        for rating in self.rTrain:
            u,i,score = rating
            uIndS = self.d*u;uIndF = self.d*(u+1);
            iIndS = self.nU*self.d + self.d*i; iIndF = self.nU*self.d + self.d*(i+1);
            hessD[uIndS:uIndF] += 2*theta[iIndS:iIndF]*np.dot(theta[iIndS:iIndF],d[uIndS:uIndF])
            hessD[iIndS:iIndF] += 2*theta[uIndS:uIndF]*np.dot(theta[uIndS:uIndF],d[iIndS:iIndF])

        return hessD/self.nR

    # calculates the gradient for the regularization objective
    def regHessD(self,theta,d):
        hessD = 2*d
        return hessD/theta.size

    # calculates the gradient for the scalarized objective
    def fHessD(self,theta,d):
        return (1-self.lambdaa)*self.errorHessD(theta,d)+self.lambdaa*self.regHessD(theta,d)

    def optimize(self):
        t0 = time()
        out = opt.minimize(self.fObj, self.theta, jac=self.fGrad, hessp=self.fHessD, method='Newton-CG', options={'xtol': 1e-3})
        self.theta=out.x
        self.objs=[self.errorObj(self.theta),self.regObj(self.theta)]
        self.obj = self.fObj(self.theta)
        self.training_time = time() - t0
        self.users = self.theta[:self.nU*self.d].reshape((self.nU,self.d))
        self.items = self.theta[self.nU*self.d:].reshape((self.nI,self.d))




class niseCand():
    def __init__(self, n0, n1,norm):
        self.n0 = n0
        self.n1 = n1
        self.imp = ((n0.objs[0]-n1.objs[0])/norm[0]*(n1.objs[1]-n0.objs[1])/norm[1])

    def calcLambdaa(self):
        self.lambdaa = np.linalg.solve([[self.n0.objs[0],self.n0.objs[1],-1],[self.n1.objs[0],self.n1.objs[1],-1],[1,1,0]],[0,0,1])[1]


def nise(ratings,nUsers,nItems,nSol=50,hVError=0.001,d=100,tol=10^-2,batchsize=None):
    init={}
    print 'Nise started: ',multiprocessing.current_process()
    
    pmf1 = mf(ratings,nUsers,nItems,d,lambdaa=0.9)
    pmf1.optimize()
    print 'Solution 0','Errors: ',pmf1.objs
    print '          ','Time: ',pmf1.training_time,'Process: ',multiprocessing.current_process(),'\n'

    pmf0 = mf(ratings,nUsers,nItems,d,lambdaa=0.0)
    pmf0.optimize()
    print 'Solution 1','Errors: ',pmf0.objs
    print '          ','Time: ',pmf0.training_time,'Process: ',multiprocessing.current_process(),'\n'

    # calculate the range between the objectives
    norm=[(pmf0.objs[0]-pmf1.objs[0]),(pmf1.objs[1]-pmf0.objs[1])]

    efList=[niseCand(pmf0,pmf1,norm)]

    out=[pmf1,pmf0]

    sols=2

    while efList!=[]:
        actual=efList.pop(0)
        #print 'Actual importance', actual.imp, 'List size',len(efList)
        if actual.imp>hVError and abs(actual.n0.getReg()-actual.n1.getReg())>tol and sols<nSol:

            actual.calcLambdaa()
            pmf = mf(ratings,nUsers,nItems,d,actual.lambdaa,out)
            pmf.optimize()

            #if numpy.linalg.norm((pmf.objs-actual.n0)/norm)>tol and numpy.linalg.norm((pmf.objs-actual.n1)/norm)>tol:

            next = niseCand(pmf,actual.n1,norm)
            efList.append(next)

            next = niseCand(actual.n0,pmf,norm)
            efList.append(next)

            print 'Solution',sols,'Errors: ',pmf.objs
            print '          ','Time: ',pmf.training_time,'Process: ',multiprocessing.current_process(),'\n'
            out.append(pmf)

            sols+=1
    return out

def plotPareto(list_,ratings):
    plotL=np.array([i.objs for i in list_])
    plt.plot(plotL[:,0],plotL[:,1],'ok')
    plt.xlabel('Squared error objective')
    plt.ylabel('Regularization objective (L2 norm)')
    plt.savefig('pareto.jpg')

def niseRun(poolpar):
    fold,d=poolpar
    train,trainU,trainI,valid,validU,validI,test,testU,testI=fold_load('ml-100k',fold)
    t0 = time()
    out=nise(train,trainU,trainI,d=d)
    plotPareto(out,train)
    with open('models/u-100k-fold-d%d-' % d +str(fold)+'.out', 'wb') as handle:
        pickle.dump(out, handle)
    runtime = time()-t0
    print 'Done: ',time()-t0,' s',multiprocessing.current_process()
    with open('models/u-100k-fold-d%d-' % d +str(fold)+'runtime.out', 'wb') as handle:
        pickle.dump(runtime, handle)



if __name__ == "__main__":
    poolList = [(fold,d) for fold in range(5) for d in [5,15,25]]
    p=multiprocessing.Pool(4)
    p.map(niseRun,poolList)
    #niseRun(poolList[0])