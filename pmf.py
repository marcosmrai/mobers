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

class ProbabilisticMatrixFactorization():

    def __init__(self, ratings, nUsers, nItems, latent_d=50,regularization_strength=0.1):
        self.latent_d = latent_d
        self.learning_rate = .001
        self.regularization_strength = regularization_strength

        self.converged = False

        self.num_users = nUsers
        self.num_items = nItems

        self.users = numpy.ones((self.num_users, self.latent_d))/numpy.sqrt(self.latent_d)*0.001
        self.items = numpy.ones((self.num_items, self.latent_d))/numpy.sqrt(self.latent_d)*0.001
        self.training_time = 0

    def updateReg(self,regularization_strength):
        self.regularization_strength=regularization_strength
        self.learning_rate = .0001
        self.converged = False

    def getReg(self):
        return self.regularization_strength

    def sampleError(self,rating,users,items):
        i,j,rat=rating
        return (i,j,numpy.dot(users[i],items[j])-rat)

    def error(self, ratings, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        #self.errorVec=[self.sampleError(rating_,users,items) for rating_ in ratings]
        self.errorVec=[(i,j,numpy.dot(users[i],items[j])-rat) for i,j,rat in ratings]

        sq_error = sum([(err)**2 for i, j, err in self.errorVec])
        L2_norm = numpy.sum(users**2)+numpy.sum(items**2)

        return (1-self.regularization_strength)*sq_error + self.regularization_strength * L2_norm

    def objErrors(self, ratings, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        sq_error = sum([(rating-numpy.dot(users[i],items[j]))**2 for i, j, rating in ratings])
        L2_norm = numpy.sum(users**2)+numpy.sum(items**2)

        return np.array([sq_error,L2_norm])

    def update(self, ratings):
        initial_lik = self.error(ratings)

        grad_o = numpy.zeros((self.num_users, self.latent_d))
        grad_d = numpy.zeros((self.num_items, self.latent_d))

        for i, j, err_ in self.errorVec:
            r_hat = numpy.dot(self.users[i],self.items[j])
            grad_o[i] += self.items[j] * (err_)
            grad_d[j] += self.users[i] * (err_)

        while (not self.converged):
            initial_lik = self.error(ratings)

            self.try_updates(grad_o, grad_d)

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
        lambd = self.regularization_strength

        self.new_users = self.users - alpha * ((1-lambd)*grad_o + lambd * self.users)
        self.new_items = self.items - alpha * ((1-lambd)*grad_d + lambd * self.items)


    def undo_updates(self):
        # Don't need to do anything here
        pass


    def print_latent_vectors(self):
        print "Users"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.users[i, d],
            print

        print "Items"
        for i in range(self.num_items):
            print i,
            for d in range(self.latent_d):
                print self.items[i, d],
            print


    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)

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
        print count
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

def nise(ratings,nUsers,nItems,nSol=100,hVError=0.001,latent_d=100,
         batchsize=None):
    init={}
    print 'Nise started: ',multiprocessing.current_process()
    pmf1 =ProbabilisticMatrixFactorization(ratings,nUsers,nItems,latent_d,regularization_strength=0.9)
    pmf1.gradient_descent(ratings, batchsize)
    print 'Solution 0','Errors: ',pmf1.objErrors(ratings),multiprocessing.current_process()

    pmf0 = copy.deepcopy(pmf1)
    pmf0.updateReg(regularization_strength=0.0)
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
        actual=efList.pop(0)
        if actual['err']>hVError and abs(actual['N1'].getReg()-actual['N2'].getReg())>10**-2 and sols<nSol:
            actual['reg']=-(actual['N1e'][0]-actual['N2e'][0])/(actual['N1e'][1]-actual['N2e'][1])
            actual['reg']=actual['reg']/(actual['reg']+1)
            alpha=np.random.random()
            actual['sol']=copy.deepcopy(actual['N2'])
            actual['sol'].updateReg(regularization_strength=actual['reg'])
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
    #plt.plot(0,0,'.')
    plt.xlabel('Squared error objective')
    plt.ylabel('Regularization objective (L2 norm)')
    #plt.show()

def niseRun(fold):
    train,trainU,trainI,valid,validU,validI,test,testU,testI=fold_load('ml-100k',fold)
    t0 = time()
    for d in [5,15,25]:
        out=nise(train,trainU,trainI,latent_d=d)
        with open('models/u-100k-fold-d%d-' % d +str(fold)+'.out', 'wb') as handle:
            pickle.dump(out, handle)
        runtime = time()-t0
        print 'Done: ',time()-t0,' s',multiprocessing.current_process()
        with open('models/u-100k-fold-d%d-' % d +str(fold)+'runtime.out', 'wb') as handle:
            pickle.dump(runtime, handle)



if __name__ == "__main__":
    '''

    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()
    ratings = numpy.array(ratings).astype(float)
    else:
        ratings=read_ratings('ml-100k/u.data')

    out=nise(ratings,latent_d=50)

    with open('u-100k.out', 'wb') as handle:
        pickle.dump((out,ratings), handle)


    with open('u-100k.out', 'rb') as handle:
        out,ratings = pickle.load(handle)
    '''
    p=multiprocessing.Pool(5)
    p.map(niseRun,range(5))
#    with open('result/u-100k-fold-'+str(fold)+'.out', 'rb') as handle:
#        out,train,valid,test=pickle.load(handle)
#    plotPareto(out,valid)
