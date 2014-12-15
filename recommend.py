import numpy as np
from bisect import insort
import os
from pickle import dump, load
from numpy.linalg import inv

class Recommender(object):

    def __init__(self, item_MF=None, threshold=3.0):
        self.item_MF = []
        self.ITpinv = []
        self.n_items = []
        if item_MF is not None:
            self.set_itemMF(item_MF)
        self.threshold = threshold


    def set_userMF(self, mat):
        self.user_MF = mat

    def set_itemMF(self, mat):
        self.item_MF = mat
        self.n_items = mat.shape[0]

    def get_user_vector(self, user_id):
        return self.user_MF[user_id, :]

    def transform_user(self, user_vector):
        #print 'item MF shape', self.item_MF.shape
        idx = user_vector[0,:]>0
        mat = self.item_MF[idx,:]
        user_vector = user_vector[0,idx]
        self.ITpinv = np.dot(mat, (inv(np.dot(mat.T, mat))))
        return np.dot(user_vector, self.ITpinv)

    def get_list(self, user_MF_vector, unrated_items, topk):
        new_ratings = []
        for item in unrated_items:
            r_hat = sum(user_MF_vector*(self.item_MF[item, :]))
            if r_hat >= self.threshold:
                insort(new_ratings, (r_hat, item))
        recommended_list = [(item, rating)
                            for rating, item in new_ratings[-topk:][::-1]]

        return recommended_list

if __name__ == '__main__':
    factors = 100
    nitems = 1000
    n_unrated = nitems-30

    user_MF_vector = np.random.random(factors)
    item_MF = np.random.random((nitems, factors))

    unrated_items = np.random.random_integers(1, nitems, n_unrated)
    topk = 10
    RecSys = Recommender(item_MF)

    r_list = RecSys.get_list(user_MF_vector, unrated_items, topk)

    print r_list
