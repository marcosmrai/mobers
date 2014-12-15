'''
Evaluation module
'''
import numpy as np
from numpy.random import choice
from ensemble import Majority, WeightedVote
from dbread import fold_load
from recommend import Recommender
from pmf import ProbabilisticMatrixFactorization
from pickle import dump, load
from multiprocessing import Pool


class Evaluation(object):
    '''
    Evaluates a recommender system.
    '''
    def __init__(self, RS, pctHidden=0.2, topk=5):
        self.RS = RS
        self.topk = topk
        self.pct_hidden = pctHidden

    def _split_positive_negative(self, user_vector):
        '''
        Split items in positive and negative evaluations
        To be used in P&R calculation
        '''
        u_positives = []
        u_negatives = []
        for i, rating in enumerate(user_vector):
            if rating >= self.RS.threshold:
                u_positives.append(i)
            elif rating >0:
                u_negatives.append(i)
        return (u_positives, u_negatives)

    def precision_recall(self, user_vector):
        u_positives, u_negatives = self._split_positive_negative(user_vector)
        #print '# + and -', len(u_positives), len(u_negatives)
        # Hide some items to check on them later
        random_pick = lambda(aList): list(
                   choice(aList,
                   np.ceil(self.pct_hidden*len(aList)),
                   replace=False)) if aList != [] else aList
        hidden_positives = random_pick(u_positives)  # u and Ihid
        hidden = hidden_positives + random_pick(u_negatives)
        #print 'hidden', len(hidden)
        new_vector = [0 if i in hidden else rating
                      for i, rating in enumerate(user_vector)]
        unrated = [i for i, rating in enumerate(new_vector) if rating == 0]
        # Transform user_vector with the curent MF and generate recomendations
        new_vector = self.RS.transform_user(np.array(new_vector, ndmin=2))
        rlist = dict(self.RS.get_list(new_vector, hidden, self.topk))

        # Calculate precision and recall
        #r and Ihid
        pred_hidden = set(hidden) & set(rlist)
        #r and u and Ihid
        pred_hidden_positives = pred_hidden & set(hidden_positives)

        #print 'pred hidden positives, pred hidden, hidden_positives ', len(pred_hidden_positives), len(pred_hidden), len(hidden_positives)

        if len(hidden_positives) > 0:
            recall = len(pred_hidden_positives)/float(len(hidden_positives))
        else:
            recall = 1.

        if len(pred_hidden) > 0:
            precision = len(pred_hidden_positives)/float(len(pred_hidden))
        elif len(u_positives) == 0:
            precision = 1.
        else:
            precision = 0.

        #print 'pr', precision, recall
        return (precision, recall)

def get_user_vectors(test, n_items):
    users = set([u for u, i, r in test])
    #print 'user set', len(users)
    vectors = []
    for user in users:
        user_vector = [0]*n_items
        for u, i, r in test:
            if user == u:
                    user_vector[i] = r

        vectors.append(user_vector)
    return vectors

def eval_users(test, evaluator, n_items):
    precision = []
    recall = []
    users = set([u for u, i, r in test])
    #print 'user set', len(users)
    for user in users:
        user_vector = [0]*n_items
        for u, i, r in test:
            if user == u:
                user_vector[i] = r

        P, R = evaluator.precision_recall(user_vector)
        #print 'pr', P, R
        precision.append(P)
        recall.append(R)

    return (precision, recall)


def performance(k, d=100, topk=10):
    print 'fold', k, d, topk

    # returns train, valid, test
    with open('data/u-100k-fold-d%d-%d.out' % (d, k), 'rb') as f:
        pmf_list = load(f)
        #pmf_list = pmf_list[:3]
    train, trainU, trainI, valid, validU, validI, test, testU, testI = \
    fold_load('data/ml-100k',k)

    print 'loaded pmf_list'


    RS_list = []
    for mf_id, pmf in enumerate(pmf_list):
        RS_list.append(Recommender(item_MF=pmf.items))
    evalu_RS_list = [Evaluation(RS=RS, topk=topk) for RS in RS_list]
    print 'RS_list created'


    n_items = RS_list[0].n_items
    result = []

    for mf_id, evaluator in enumerate(evalu_RS_list):
        P, R = eval_users(valid, evaluator, n_items)
        result.append([d,
                       pmf_list[mf_id].regularization_strength,
                       np.mean(P), np.mean(R)])
        print '!!!concluded RS ', mf_id, 'PR', np.mean(P), np.mean(R)


    # ensembles
    E1 = Majority(RS_list, threshold=3)
    precisions = [line[2] for line in result]
    E2 = WeightedVote(RS_list, weights=precisions, threshold=3)
    best_RS = np.argmax(precisions)
    E3 = RS_list[best_RS]
    evalu_ensemble = [Evaluation(RS=E1, topk=topk),
                      Evaluation(RS=E2, topk=topk),
                      Evaluation(RS=E3, topk=topk)]
    print 'ensembles created'

    for e_id, evaluator in enumerate(evalu_ensemble):
        P, R = eval_users(test, evaluator, n_items)
        P = np.mean(P)
        R = np.mean(R)

        result.append([d, P, R])
        print '!!!concluded E', e_id, 'PR', P, R

    result[-1].insert(1, pmf_list[best_RS].regularization_strength)

    print 'saving results'

    with open('results/u-100k-fold-%d-d%d-top%d-results.out' % (d, k, topk), 'wb') as f:
        dump(result, f)

def pool_performance(tup):
    k, d, topk = tup
    performance(k=k, d=d, topk=topk)

if __name__ == '__main__':
    #performance(k=0, d=50, topk=10)
    pool_args = [(k, d, topk) for k in range(5)
              for d in [50,100]
              for topk in range(5, 20+1, 5)]
    p = Pool(4)
    p.map(pool_performance, pool_args)
    #performance(d=50)
