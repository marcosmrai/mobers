'''
Evaluation module
'''
import numpy as np
from numpy.random import choice
from ensemble import Majority, WeightedVote, FilteredMajority
import dbread
from dbread import fold_load
from recommend import Recommender
from pmf import mf
from pickle import dump, load
from multiprocessing import Pool

'''
GLOBAL PARAMETERS

Parameter of the main script
'''
NFOLDS = 5
DIMLIST = [5, 15, 25, 50, 100]
TOPKLIST = [5]
PCTHIDDEN = 0.2
THRESHOLD = 3
MODELFOLDER = 'models/'
RESULTFOLDER = 'results/'
GEN_HIDDEN_ITEMS = False # If true will ramdomly sample and save the hidden items for test and validation
NPROC = 1 # if >1, will create a pool with NPROC workers


ENSEMBLE_ORDER = {'vote': -4, 'filtered': -3, 'weighted': -2,  'best': -1}
METRIC_ID = {'p':1, 'r':2}

class Evaluation(object):
    '''
    Evaluates a recommender system.
    '''
    def __init__(self, RS, pctHidden=0.2, topk=5):
        self.RS = RS
        self.topk = topk
        self.pct_hidden = pctHidden

    def precision_recall(self, user_vector, hidden):
        hidden_positives, hidden_negatives = hidden
        hidden = hidden_positives + hidden_negatives
        u_positives, u_negatives = \
            split_positive_negative(user_vector, threshold = self.RS.threshold)
        new_vector = [0 if i in hidden else rating
                      for i, rating in enumerate(user_vector)]
        unrated = [i for i, rating in enumerate(new_vector) if rating == 0]
        # Transform user_vector with the curent MF and generate recomendations
        new_vector = self.RS.transform_user(np.array(new_vector, ndmin=2))
        rlist = dict(self.RS.get_list(new_vector, hidden, self.topk))
        #print rlist
        # Calculate precision and recall
        #r and Ihid
        pred_hidden = set(hidden) & set(rlist)
        #r and u and Ihid
        pred_hidden_positives = pred_hidden & set(hidden_positives)

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

        return (precision, recall)

def get_user_vectors(test, n_items):
    users = set([u for u, i, r in test])
    #print 'user set', len(users)
    vectors = {}
    for user in users:
        user_vector = [0]*n_items
        for u, i, r in test:
            if user == u:
                user_vector[i] = r

        vectors[user] = user_vector
    return vectors

def split_positive_negative(user_vector, threshold=3):
    '''
    Split items in positive and negative evaluations
    To be used in P&R calculation
    '''
    u_positives = []
    u_negatives = []
    for i, rating in enumerate(user_vector):
        if rating >= threshold:
            u_positives.append(i)
        elif rating >0:
            u_negatives.append(i)
    return (u_positives, u_negatives)


def eval_users(test, hidden, evaluator, n_items):
    '''
    test: list with test tuples (user, item, rating)
    hidden: dictionary with list of hidden test items per user key
    evaluator: Evaluator object
    n_items: total number of items in the database
    '''
    precision = []
    recall = []
    users = set([u for u, i, r in test])
    #print 'user set', len(users)
    for user in users:
        user_vector = [0]*n_items
        for u, i, r in test:
            if user == u:
                user_vector[i] = r

        P, R = evaluator.precision_recall(user_vector, hidden[user])
        #print 'pr', P, R
        precision.append(P)
        recall.append(R)

    return (precision, recall)


def get_hidden_items(u_positives, u_negatives, pct_hidden):
    popular = dbread.loadFreqItems('ml-100k')
    remove_popular = lambda alist: \
                     [item for item in alist
                      if item not in popular]
    u_positives = remove_popular(u_positives)
    u_negatives = remove_popular(u_negatives)
    # Hide some items to check on them later
    random_pick = lambda(aList): list(
               choice(aList,
               np.ceil(pct_hidden*len(aList)),
               replace=False)) if aList != [] else aList
    hidden_positives = random_pick(u_positives)  # u and Ihid
    return (hidden_positives, random_pick(u_negatives))


def save_test_items(pct_hidden = 0.2, threshold=3):
    '''
    Saves a set of hidden items for each user in the validation and test set
    '''


    for k in range(NFOLDS):
        # Load fold k
        train, trainU, trainI, valid, validU, validI, test, testU, testI = \
            fold_load('ml-100k',k)
        n_items = testI
        hidden = {}
        user_vectors = get_user_vectors(valid, n_items)
        for user in user_vectors:
            u_positives, u_negatives = \
                split_positive_negative(user_vectors[user], threshold)
            hidden[user] = get_hidden_items(u_positives, u_negatives, pct_hidden)
        with open('ml-100k/fold%d-hidden.pickle'%k, 'wb') as f:
            dump(hidden, f)
    # repeat for test set
    hidden = {}
    user_vectors = get_user_vectors(test, n_items)
    for user in user_vectors:
        u_positives, u_negatives = \
            split_positive_negative(user_vectors[user], threshold)

        hidden[user] = get_hidden_items(u_positives, u_negatives, pct_hidden)
        with open('ml-100k/test-hidden.pickle', 'wb') as f:
            dump(hidden, f)

def performance(k, d, topk=5):
    '''
    k: fold index
    d: latent dimensionality of pmf model
    topk: size of recomendation list
    '''

    print 'fold %d, dim %d, top %d list' % (k, d, topk)

    # returns train, valid, test
    with open(MODELFOLDER + '/u-100k-fold-d%d-%d.out' % (d, k), 'rb') as f:
        pmf_list = load(f)
        #pmf_list = pmf_list[:3]
    train, trainU, trainI, valid, validU, validI, test, testU, testI = \
        fold_load('ml-100k',k)

    with open('ml-100k/fold%d-hidden.pickle'%k, 'rb') as f:
        hidden_v = load(f)
    with open('ml-100k/test-hidden.pickle', 'rb') as f:
        hidden_t = load(f)

    print 'loaded pmf_list'

    RS_list = []
    for mf_id, pmf in enumerate(pmf_list):
        RS_list.append(Recommender(item_MF=pmf.items))
    evalu_RS_list = [Evaluation(RS=RS, topk=topk) for RS in RS_list]
    print 'RS_list created'
    n_items = RS_list[0].n_items


    result = []

    for mf_id, evaluator in enumerate(evalu_RS_list):
        P, R = eval_users(valid, hidden_v, evaluator, n_items)
        result.append([d,
                       pmf_list[mf_id].lambdaa,
                       np.mean(P), np.mean(R)])
        print '!!!concluded RS ', mf_id, 'PR', np.mean(P), np.mean(R)


    # ensembles
    # If order is changed, please adjust ENSMEBLE_ORDER constant
    E1 = Majority(RS_list, threshold=3)
    precisions = [line[2] for line in result]
    E1f = FilteredMajority(RS_list, performances=precisions, threshold=3)
    E2 = WeightedVote(RS_list, weights=precisions, threshold=3)
    best_RS = np.argmax(precisions)
    E3 = RS_list[best_RS]
    evalu_ensemble = [Evaluation(RS=E1, topk=topk),
                      Evaluation(RS=E1f, topk=topk),
                      Evaluation(RS=E2, topk=topk),
                      Evaluation(RS=E3, topk=topk)]
    print 'ensembles created'

    for e_id, evaluator in enumerate(evalu_ensemble):
        P, R = eval_users(test, hidden_t, evaluator, n_items)
        P = np.mean(P)
        R = np.mean(R)

        result.append([d, P, R])
        print '!!!concluded E', e_id, 'PR', P, R

    result[-1].insert(1, pmf_list[best_RS].lambdaa)

    print 'saving results'

    times = [pmf.training_time for pmf in pmf_list]
    result_folder = RESULTFOLDER
    with open(result_folder+'u-100k-fold-%d-d%d-top%d-times.out' % (d, k, topk), 'wb') as f:
        dump(times, f)
    with open(result_folder+'u-100k-fold-%d-d%d-top%d-results.out' % (d, k, topk), 'wb') as f:
        dump(result, f)

def ensembles_performance(k, d, topk=5):
    '''
    k: fold index
    d: latent dimensionality of pmf model
    topk: size of recomendation list
    '''

    print 'fold %d, dim %d, top %d list' % (k, d, topk)

    # returns train, valid, test
    with open(MODELFOLDER + '/u-100k-fold-d%d-%d.out' % (d, k), 'rb') as f:
        pmf_list = load(f)
        #pmf_list = pmf_list[:3]
    train, trainU, trainI, valid, validU, validI, test, testU, testI = \
        fold_load('ml-100k',k)

    with open('ml-100k/fold%d-hidden.pickle'%k, 'rb') as f:
        hidden_v = load(f)
    with open('ml-100k/test-hidden.pickle', 'rb') as f:
        hidden_t = load(f)

    print 'loaded pmf_list'

    RS_list = []
    for mf_id, pmf in enumerate(pmf_list):
        RS_list.append(Recommender(item_MF=pmf.items))
    print 'RS_list created'
    n_items = RS_list[0].n_items

    with open(RESULTFOLDER+'u-100k-fold-%d-d%d-top%d-results.out' % (d, k, topk), 'rb') as f:
        result = load(f)

    result = result[0:len(RS_list)]
    # ensembles
    # If order is changed, please adjust ENSMEBLE_ORDER constant
    E1 = Majority(RS_list, threshold=3)
    precisions = [line[2] for line in result]
    E1f = FilteredMajority(RS_list, performances=precisions, threshold=3)
    E2 = WeightedVote(RS_list, weights=precisions, threshold=3)
    best_RS = np.argmax(precisions)
    E3 = RS_list[best_RS]
    evalu_ensemble = [Evaluation(RS=E1, topk=topk),
                      Evaluation(RS=E1f, topk=topk),
                      Evaluation(RS=E2, topk=topk),
                      Evaluation(RS=E3, topk=topk)]
    print 'ensembles created'

    for e_id, evaluator in enumerate(evalu_ensemble):
        P, R = eval_users(test, hidden_t, evaluator, n_items)
        P = np.mean(P)
        R = np.mean(R)

        result.append([d, P, R])
        print '!!!concluded E', e_id, 'PR', P, R

    result[-1].insert(1, pmf_list[best_RS].lambdaa)

    print 'saving results'

    with open(RESULTFOLDER+'u-100k-fold-%d-d%d-top%d-results.out' % (d, k, topk), 'wb') as f:
        dump(result, f)

def pool_performance(tup):
    k, d, topk = tup
    performance(k=k, d=d, topk=topk)

if __name__ == '__main__':
    if GEN_HIDDEN_ITEMS:
        save_test_items(PCTHIDDEN, THRESHOLD)
    pool_args = [(k, d, topk)
                 for k in range(NFOLDS)
                 for d in DIMLIST
                 for topk in TOPKLIST]
    if NPROC > 1:
        p = Pool(1)
        p.map(pool_performance, pool_args)
    else:
        map(pool_performance, pool_args)


