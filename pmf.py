import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy
import pickle

from dbread import *

class ProbabilisticMatrixFactorization():

    def __init__(self, rating_tuples, latent_d=1):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.regularization_strength = 0.1
        
        self.ratings = numpy.array(rating_tuples).astype(float)
        self.converged = False

        self.num_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.num_items = int(numpy.max(self.ratings[:, 1]) + 1)
        
        self.users = numpy.random.random((self.num_users, self.latent_d))
        self.items = numpy.random.random((self.num_items, self.latent_d))



    def likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
            
        sq_error = sum([(rating-numpy.dot(users[i],items[j]))**2 for i, j, rating in self.ratings])
        L2_norm = numpy.sum(users**2)+numpy.sum(items**2)

        return -sq_error - self.regularization_strength * L2_norm
        
        
    def update(self):

        grad_o = numpy.zeros((self.num_users, self.latent_d))
        grad_d = numpy.zeros((self.num_items, self.latent_d))        

        for i, j, rating in self.ratings:
            r_hat = numpy.sum(self.users[i] * self.items[j])     
            grad_o[i] += self.items[j] * (rating - r_hat)
            grad_d[j] += self.users[i] * (rating - r_hat)

        while (not self.converged):
            initial_lik = self.likelihood()

            print "  setting learning rate =", self.learning_rate
            self.try_updates(grad_o, grad_d)

            final_lik = self.likelihood(self.new_users, self.new_items)

            if final_lik > initial_lik:
                self.apply_updates(grad_o, grad_d)
                self.learning_rate *= 1.25

                if final_lik - initial_lik < .1:
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
        beta = -self.regularization_strength

        self.new_users = self.users + alpha * (beta * self.users + grad_o)
        self.new_items = self.items + alpha * (beta * self.items + grad_d)
        

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


if __name__ == "__main__":

    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()

    #plot_ratings(ratings)

    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=5)
    
    liks = []
    while (not pmf.update()):
        lik = pmf.likelihood()
        liks.append(lik)
        print "L=", lik
        pass
    
    plt.figure()
    plt.plot(liks)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")

    plot_latent_vectors(pmf.users, pmf.items)
    plot_predicted_ratings(pmf.users, pmf.items)
    plt.show()

    pmf.print_latent_vectors()
    pmf.save_latent_vectors("models/")
