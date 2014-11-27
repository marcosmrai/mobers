'''
Evaluation module
'''
from recommend import Recommender
import numpy as np
from np.random import choice


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


class Kfold(object):
    '''
    Performs k fold testing or validation
    '''
    def __init__(self, k, RS):
        self.k = k
        self.evaluation = Evaluation(RS)
        self.folds = []

    def set_system(self, RS):
        self.evaluation = RS

    def gen_split(self, n_users):
        '''
        Generate k-fold split to be used in k-fold validation
        '''
        quantity = n_users/self.k
        idx = np.random.permutation(range(n_users))
        for i in xrange(0, self.k-1):
            self.folds.append(idx[i*quantity:(i+1)*quantity])
        self.folds.append(idx[self.k-1*quantity:])

    def _pick_split(self, val_fold):
        '''
        returns indexes for training and validation, with fold[val_fold] being
        the validation fold
        '''
        training = []
        for i in range(k):
            if i == val_fold:
                validation = self.fold[val_fold]
            else:
                training += self.fold[val_fold]
        return (training, validation)

    def run(self):
        '''
        Do k-fold validation
        '''
        #TODO implement
        pass
