import numpy as np
from bisect import insort


class Recommender(object):

    def __init__(self, item_MF=None, threshold=3.0):
        self.item_MF = []
        self.ITpinv = []
        self.n_items = []
        if item_MF is not None:
            self.set_itemMF(item_MF)
        self.last_user_query = []
        self.threshold = threshold
        self.last_recommendation = []

    def set_userMF(self, mat):
        self.user_MF = mat

    def set_itemMF(self, mat):
        self.item_MF = mat
        self.ITpinv = np.dot(mat, (np.linalg.inv(np.dot(mat.T, mat))))
        self.n_items = mat.shape[0]

    def get_user_vector(self, user_id):
        return self.user_MF[user_id, :]

    def transform_user(self, user_vector):
        return np.dot(user_vector, self.ITpinv)

    def get_list(self, user_MF_vector, unrated_items, topk):
        if self.last_user_query != (user_MF_vector, unrated_items):
            self.last_user_query = (user_MF_vector, unrated_items)
            new_ratings = []
            for item in unrated_items:
                r_hat = sum(user_MF_vector*(self.item_MF[item-1, :]))
                if r_hat >= self.threshold:
                    insort(new_ratings, (r_hat, item))
            recommended_list = [(item, rating)
                                for rating, item in new_ratings[-topk:][::-1]]:
        else:
            recommended_list = self.last_recommendation
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
