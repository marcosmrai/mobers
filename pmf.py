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

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib
#import matplotlib.pyplot as plt

class mf():
    def __init__(self, rTrain, nUsers, nItems, d=50, lambdaa=0.1): 
        self.d = d
        self.lambdaa = lambdaa

        self.rTrain = rTrain

        self.nU = nUsers
        self.nI = nItems
        self.nR = len(rTrain)

        self.theta = numpy.ones(self.nU*self.d + self.nI*self.d)/numpy.sqrt(self.d)#*0.001
        self.training_time = 0

        # create the inicialization function

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
	    #print rating
            obj+=self.sampleError(rating,theta)**2

        return obj/self.nR

    # calculates the regularization objective
    def regObj(self,theta):
        return theta.sum()/(self.nU*self.d + self.nI*self.d)

    # calculates the scalarized objective 
    def fObj(self,theta):
	print numpy.dot(theta,theta)
        print self.errorObj(theta),self.regObj(theta),(1-self.lambdaa)*self.errorObj(theta)+self.lambdaa*self.regObj(theta)
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
        grad = 2*theta/(self.nU*self.d + self.nI*self.d)
        return grad

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
        hessD = 2*d/(self.nU*self.d + self.nI*self.d)
        return hessD

    # calculates the gradient for the scalarized objective
    def fHessD(self,theta,d):
        return (1-self.lambdaa)*self.errorHessD(theta,d)+self.lambdaa*self.regHessD(theta,d)

    def optimize(self):
        t0 = time()
        #out = opt.minimize(self.fObj, self.theta)
        out = opt.minimize(self.fObj, self.theta, jac=self.fGrad, method='Newton-CG',hessp=self.fHessD,options={'xtol': 1e-3})
        self.theta=out.x
        self.objs=[self.errorObj(self.theta),self.regObj(self.theta)]
        self.obj = self.fObj(self.theta)
        self.training_time = time() - t0


class ProbabilisticMatrixFactorization():

    def __init__(self, ratings, nUsers, nItems, d=50,lambdaa=0.1):
        self.d = d
        self.learning_rate = .001
        self.lambdaa = lambdaa

        self.converged = False

        self.num_users = nUsers
        self.num_items = nItems
        self.num_ratings = len(ratings)

        self.users = numpy.ones((self.num_users, self.d))/numpy.sqrt(self.d)#*0.001
        self.items = numpy.ones((self.num_items, self.d))/numpy.sqrt(self.d)#*0.001
        self.training_time = 0

    # update the reg for a new model
    def updateReg(self,lambdaa):
        self.lambdaa=lambdaa
        self.learning_rate = .0001
        self.converged = False

    # return the regularization parameter
    def getReg(self):
        return self.lambdaa

    # calculates the error of a rating
    def sampleError(self,rating,users,items):
        i,j,rat=rating
        return (i,j,numpy.dot(users[i],items[j])-rat)

    # calculate the scalarized error
    def error(self, ratings, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        # update the error vector
        self.errorVec=[(i,j,numpy.dot(users[i],items[j])-rat) for i,j,rat in ratings]

        # calculate the objective functions
        sq_error = sum([(err)**2 for i, j, err in self.errorVec])/self.num_ratings
        L2_norm = numpy.sum(users**2)/users.size+numpy.sum(items**2)/items.size

        #return the scalarized objective error
        return (1-self.lambdaa)*sq_error + self.lambdaa * L2_norm

    # calculate the objective functions
    def objErrors(self, ratings, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        sq_error = sum([(rating-numpy.dot(users[i],items[j]))**2 for i, j, rating in ratings])/self.num_ratings
        L2_norm = numpy.sum(users**2)/users.size+numpy.sum(items**2)/items.size

        return np.array([sq_error,L2_norm])

    # apply an update
    def update(self, ratings):
	initial_lik = self.error(ratings)
	#print initial_lik, self.objErrors(ratings)
        # inicializate the gradient with the regularization factor
        grad_o = self.lambdaa*2*self.users/self.users.size
        grad_d = self.lambdaa*2*self.items/self.items.size

        # add to the gradient the error factor
        for i, j, err_ in self.errorVec:
            grad_o[i] += (1-self.lambdaa)*self.items[j] * (err_)
            grad_d[j] += (1-self.lambdaa)*self.users[i] * (err_)

        while (not self.converged):
            #calculates the initial likelehood
            initial_lik = self.error(ratings)

            self.try_updates(grad_o, grad_d)

            #calculates the updated likelehood
            final_lik = self.error(ratings,self.new_users, self.new_items)

            if final_lik < initial_lik:
                self.apply_updates(grad_o, grad_d)
                self.learning_rate *= 1.25

                if initial_lik - final_lik < .001:
                    self.converged = True

                break
            else:
                self.learning_rate *= .5
                self.undo_updates()

            if self.learning_rate < 1e-10:
                self.converged = True

        return self.converged


    def apply_updates(self, grad_o, grad_d):
        self.users = numpy.copy(self.new_users)
        self.items = numpy.copy(self.new_items)


    def try_updates(self, grad_o, grad_d):
        alpha = self.learning_rate
        lambd = self.lambdaa

        self.new_users = self.users - alpha * ((1-lambd)*grad_o + lambd * self.users)
        self.new_items = self.items - alpha * ((1-lambd)*grad_d + lambd * self.items)


    def undo_updates(self):
        # Don't need to do anything here
        pass


    def print_latent_vectors(self):
        print "Users"
        for i in range(self.num_users):
            print i,
            for d in range(self.d):
                print self.users[i, d],
            print

        print "Items"
        for i in range(self.num_items):
            print i,
            for d in range(self.d):
                print self.items[i, d],
            print


    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.d)
        self.items.dump(prefix + "%sd_items.pickle" % self.d)

    def gradient_descent(self, ratings, batchsize=None):
        t0 = time()
        converged = False
        if batchsize != None:
            shuffle(ratings)
            idx = range(0, len(ratings)-batchsize, batchsize) + [len(ratings)-1]
            n = len(idx) - 1
            self.learning_rate /= float(n)
            i = 0
        count = 0
        while not converged:
            if batchsize != None:
                batch = ratings[idx[i]:idx[i+1]]
                if (i+1) % n < i:
                    shuffle(ratings)
                    self.learning_rate = min(1e-4/float(n),self.learning_rate)
                else: self.learning_rate *= 0.5
                i = (i+1) % n
            else: batch = ratings
            converged = self.update(batch)
            count += 1
            if batchsize != None:
                if count < n: converged = False
        self.objs=self.objErrors(ratings)
        self.obj = self.error(ratings)
        self.training_time = time() - t0


def plot_ratings(ratings):
    xs = []
    ys = []

    for i in range(len(ratings)):
        xs.append(ratings[i][1])
        ys.append(ratings[i][2])

    pylab.plot(xs, ys, 'bx')
    pylab.show()


def plot_latent_vectors(U, V):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cmap = cm.jet
    ax.imshow(U, cmap=cmap, interpolation='nearest')
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(V, cmap=cmap, interpolation='nearest')
    plt.title("Items")
    plt.axis("off")

def plot_predicted_ratings(U, V):
    r_hats = -5 * numpy.ones((U.shape[0] + U.shape[1] + 1,
                              V.shape[0] + V.shape[1] + 1))

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            r_hats[i + V.shape[1] + 1, j] = U[i, j]

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            r_hats[j, i + U.shape[1] + 1] = V[i, j]

    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            r_hats[i + U.shape[1] + 1, j + V.shape[1] + 1] = numpy.dot(U[i], V[j]) / 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(r_hats, cmap=cm.gray, interpolation='nearest')
    plt.title("Predicted Ratings")
    plt.axis("off")

def nise(ratings,nUsers,nItems,nSol=100,hVError=0.001,d=100,
         batchsize=None):
    init={}
    print 'Nise started: ',multiprocessing.current_process()
    pmf1 = ProbabilisticMatrixFactorization(ratings,nUsers,nItems,d,lambdaa=0.9)
    pmf1.gradient_descent(ratings, batchsize)
    print 'Solution 0','Errors: ',pmf1.objErrors(ratings),multiprocessing.current_process()

    pmf0 = copy.deepcopy(pmf1)
    pmf0.updateReg(lambdaa=0.0)
    pmf0.gradient_descent(ratings, batchsize)
    print 'Solution 1','Errors: ',pmf0.objErrors(ratings),multiprocessing.current_process()

    init={'N1':pmf1,'N1e':pmf1.objErrors(ratings),'N2':pmf0,'N2e':pmf0.objErrors(ratings)}
    norm=[(init['N2e'][0]-init['N1e'][0]),(init['N2e'][1]-init['N1e'][1])]
    init['err']=((init['N2e'][0]-init['N1e'][0])/norm[0]*(init['N2e'][1]-init['N1e'][1])/norm[1])
    efList=[init]

    out=[pmf1,pmf0]

    sols=2

    #return out

    while efList!=[] and sols<50:
	print len(efList)
        actual=efList.pop(0)
        if actual['err']>hVError and abs(actual['N1'].getReg()-actual['N2'].getReg())>10**-2 and sols<nSol:
            actual['reg']=-(actual['N1e'][0]-actual['N2e'][0])/(actual['N1e'][1]-actual['N2e'][1])
            actual['reg']=actual['reg']/(actual['reg']+1)
            alpha=np.random.random()
            actual['sol']=copy.deepcopy(actual['N2'])
            actual['sol'].updateReg(lambdaa=actual['reg'])
            actual['sol'].gradient_descent(ratings, batchsize)

            print 'Solution',sols,'Errors: ',actual['sol'].objErrors(ratings),multiprocessing.current_process()
            out.append(actual['sol'])

            next={'N1':actual['sol'],'N1e':actual['sol'].objErrors(ratings),'N2':actual['N2'],'N2e':actual['N2e']}
            next['err']=((next['N2e'][0]-next['N1e'][0])/norm[0]*(next['N2e'][1]-next['N1e'][1])/norm[1])

            efList.append(next)

            next={'N1':actual['N1'],'N1e':actual['N1e'],'N2':actual['sol'],'N2e':actual['sol'].objErrors(ratings)}
            next['err']=((next['N2e'][0]-next['N1e'][0])/norm[0]*(next['N2e'][1]-next['N1e'][1])/norm[1])

            efList.append(next)

            sols+=1
    return out

def plotPareto(list_,ratings):
    plotL=np.array([i.objErrors(ratings) for i in list_])
    plt.plot(plotL[:,0],plotL[:,1],'ok')
    plt.savefig('pareto.jpg')
    #plt.plot(0,0,'.')
    plt.xlabel('Squared error objective')
    plt.ylabel('Regularization objective (L2 norm)')
    #plt.show()

def niseRun(poolpar):
    fold,d=poolpar
    train,trainU,trainI,valid,validU,validI,test,testU,testI=fold_load('ml-100k',fold)
    t0 = time()
    #for d in [5,15,25]:
    out=nise(train,trainU,trainI,d=d)
    plotPareto(out,train)
    #with open('models/u-100k-fold-d%d-' % d +str(fold)+'.out', 'wb') as handle:
    #    pickle.dump(out, handle)
    runtime = time()-t0
    #print 'Done: ',time()-t0,' s',multiprocessing.current_process()
    #with open('models/u-100k-fold-d%d-' % d +str(fold)+'runtime.out', 'wb') as handle:
    #    pickle.dump(runtime, handle)



if __name__ == "__main__":
    '''

    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()
    ratings = numpy.array(ratings).astype(float)
    else:
        ratings=read_ratings('ml-100k/u.data')

    out=nise(ratings,d=50)

    with open('u-100k.out', 'wb') as handle:
        pickle.dump((out,ratings), handle)


    with open('u-100k.out', 'rb') as handle:
        out,ratings = pickle.load(handle)
    '''
    poolList = [(fold,d) for fold in range(5) for d in [5,15,25]]
    #p=multiprocessing.Pool(4)
    #p.map(niseRun,poolList)
    niseRun(poolList[0])
    '''
    train,trainU,trainI,valid,validU,validI,test,testU,testI=fold_load('ml-100k',fold)
    pmf2 = mf(train,trainU,trainI,5,lambdaa=0.9)
    pmf2.optimize()
    print pmf2.obj,pmf2.objs,pmf2.training_time
    pmf1 =ProbabilisticMatrixFactorization(train,trainU,trainI,5,lambdaa=0.9)
    pmf1.gradient_descent(train, None)
    print pmf1.obj,pmf1.objs,pmf1.training_time
    '''
