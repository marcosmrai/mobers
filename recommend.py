import numpy as np
from bisect import insort

def recommend(unrated_items, user_MF_vector, item_MF, topk):
    n_items = item_MF.shape[0]
    new_ratings = []
    for item in unrated_items:
        r_hat = sum(user_MF_vector*(item_MF[item-1,:]))
        insort(new_ratings, (r_hat, item))
    recommended_list = [(item,rating) for rating, item in new_ratings[-topk:][::-1]]      
    return recommended_list
    
if __name__=='__main__':
    factors = 100
    nitems = 1000
    n_unrated = nitems-30
    
    user_MF_vector = np.random.random(factors)
    item_MF = np.random.random((nitems, factors))
    
    unrated_items = np.random.random_integers(1,nitems, n_unrated)
    topk = 10  
    
    r_list = recommend(unrated_items, user_MF_vector, item_MF, topk)
    
    print r_list