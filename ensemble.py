from bisect import insort
from pmf import nise
from dbread import read_ratings, fake_ratings
from recommend import Recommender
from pmf import ProbabilisticMatrixFactorization as PMF
import numpy as np


class Ensemble(object):
    def __init__(self, RS_list, threshold=3):
        self.RS_list = RS_list
        self.r_lists = []
        self.threshold = threshold
        self.n_items =self.RS_list[0].n_items

    def transform_user(self, user_vector):
        return user_vector



class Majority(Ensemble):
    def __init__(self, RS_list, threshold=3):
        Ensemble.__init__(self, RS_list, threshold)

    def get_list(self, user_vector, unrated_items, topk):
        item_vote = [0]*self.RS_list[0].n_items
        for i, RS in enumerate(self.RS_list):
            if not np.isscalar(user_vector):
                t_user_vector = RS.transform_user(user_vector)
            else: t_user_vector = user_vector
            r_list = RS.get_list(t_user_vector, unrated_items, topk)
            self.r_lists.append(r_list)
            for item, rating in r_list:
                item_vote[item] += 1.0
        ensemble = []
        for item, votes in enumerate(item_vote):
            insort(ensemble, (votes, item))

        return [(item, votes) for votes, item in ensemble[-topk:][::-1]]


class WeightedVote(Ensemble):
    def __init__(self, RS_list, weights, threshold=3):
        Ensemble.__init__(self, RS_list, threshold)
        self.weights = weights

    def set_weights(self, weights):
        self.weights = weights

    def get_list(self, user_vector, unrated_items, topk):
        item_vote = [0]*self.RS_list[0].n_items
        for RS_id, RS in enumerate(self.RS_list):
            if not np.isscalar(user_vector):
                t_user_vector = RS.transform_user(user_vector)
            else: t_user_vector = user_vector
            r_list = RS.get_list(t_user_vector, unrated_items, topk)
            self.r_lists.append(r_list)
            for item, rating in r_list:
                item_vote[item] += self.weights[RS_id]
        ensemble = []
        for item, votes in enumerate(item_vote):
            insort(ensemble, (votes, item))
        return [(item, votes) for votes, item in ensemble[-topk:][::-1]]

if __name__ == "__main__":

    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, u, i) = fake_ratings()
        ratings = np.array(ratings).astype(float)
        MF_list = []
        for lambdar in [0.0, 0.5, 1.0]:
            pmf = PMF(ratings, latent_d=5, regularization_strength=lambdar)
            pmf.gradient_descent(ratings)
            MF_list.append(pmf)
    else:
        filename = 'ml-100k/u.data'
        ratings = read_ratings(filename)
        ratings = np.array(ratings).astype(float)
        MF_list = nise(ratings)


    ens = Majority(MF_list)
    uid = 50.0
    #gerar usuario teste
    nitems = MF_list[0].num_items
    user_vector = np.zeros((1,nitems),dtype=float)
    for line in ratings:
        if line[0] == uid:
            user_vector[:,line[1]]=line[2]
    unrated_items = [i for i, r in enumerate(user_vector) if r[0] == 0.0]
    topk = 5
    #gera lista de rec.
    rlist = ens.get_list(user_vector, unrated_items, topk)
    print 'final rlist', rlist
