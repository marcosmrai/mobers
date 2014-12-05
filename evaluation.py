'''
Evaluation module
'''
from recommend import Recommender
import numpy as np
from np.random import choice
from ensemble import Majority

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
            else:
                u_negatives.append(i)
        return (u_positives, u_negatives)

    def precision_recall(self, user_vector):
        u_positives, u_negatives = self._split_positive_negative(user_vector)
        # Hide some items to check on them later
        random_pick = lambda(aList):\
            choice(aList,
                   np.ceil(self.pct_hidden*len(aList)),
                   replace=False)
        hidden_positives = random_pick(u_positives)  # u and Ihid
        hidden = hidden_positives + random_pick(u_negatives)
        new_vector = [0 if i in hidden else rating
                      for i, rating in enumerate(user_vector)]

        # Transform user_vector with the curent MF and generate recomendations
        new_vector = self.RS.transform_user(new_vector)
        rlist = dict(self.RS.get_list(new_vector, hidden, self.topk))

        # Calculate precision and recall
        #r and Ihid
        pred_hidden = len(set(hidden) & set(rlist))
        #r and u and Ihid
        pred_hidden_positives = len(pred_hidden & hidden_positives)

        if hidden_positives > 0:
            recall = pred_hidden_positives/hidden_positives
        else:
            recall = 1

        if pred_hidden > 0:
            precision = pred_hidden_positives/pred_hidden
        elif len(u_positives) == 0:
            precision = 1
        else:
            precision = 0

        return (precision, recall)


def avg_eval(train, evaluator):
    precision = 0
    recall = 0
    for user in train:
        P, R = evaluator.precision_recall(user)
        precision += P
        recall += R
    precision /= len(train)
    recall /= len(train)
    return (precision, recall)

def performance():
    # list of PR matrices for each fold.
    # fist line is ensemble, others are nise solutions
    PR_per_fold = []
    for k in range(10):
#TODO change to prototypes of real read functions when they're implemented
        train, test = read_fold(k)
        pmf_list = read_nise_results()

        ensemble = Majority(pmf_list)

        evalu_ensemble = Evaluation(ensemble)
        evalu_RS_list = [Evaluation(RS) for RS in ensemble.RS]
        evaluators = [evalu_ensemble] + evalu_RS_list
        PR_per_fold.append(np.array([avg_eval(train, evalu) for evalu in evaluators]))
    return PR_per_fold