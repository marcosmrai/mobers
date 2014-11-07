from bisect import insort

def majority_ensemble (user_id, MF_list, topk):
    n_items = MF_list[0][1].shape[0]
    item_vote = [0]*n_items
    for user_MF, item_MF in MF_list:
        r_list = recommend(unrated_items, user_MF[user_id,], item_MF, topk)
        r_list = [item for item, rating in r_list]
        for item in r_list:
            item_vote[item]+=1
    ensemble = []
    for item,votes in enumerate (item_vote):
        insort(ensemble, (item,votes))
    return ensemble[-topk:][::-1]
    

    

