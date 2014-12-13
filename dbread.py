import numpy as np
import math
import random
import pickle

def fake_ratings(noise=.25):
    u = []
    v = []
    ratings = []

    num_users = 100
    num_items = 100
    num_ratings = 30
    latent_dimension = 10

    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * np.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * np.random.randn(latent_dimension))

    # Get num_ratings ratings per user.
    for i in range(num_users):
        items_rated = np.random.permutation(num_items)[:num_ratings]

        for jj in range(num_ratings):
            j = items_rated[jj]
            rating = np.sum(u[i] * v[j]) + noise * np.random.randn()

            ratings.append((i, j, rating))  # thanks sunquiang

    return (ratings, u, v)

def read_ratings(filename):
    with open(filename, 'r') as f:
        # base is in format: user_ID  item_ID  rating (tab separated)
        ratings = [tuple([int(elem) for elem in line.split('\t')[0:-1]]) \
                   for line in f]
       # convert indexing to 0-index
        #ratings = [(u-1,i-1,r) for u,i,r in ratings]
	ratings = np.array([[u-1,i-1,r] for u,i,r in ratings])
    return ratings

def read_user_ratings(filename, user_id):
    with open(filename, 'r') as f:
        user_ratings = [(int(line.split()[1]), int(line.split()[2]))\
                       for line in f if int(line.split()[0])==user_id]
    return user_ratings

def k_fold_gen(folder,filename,finfo,k):
    ratings=read_ratings(filename)
    ratings.shape
    with open(finfo, 'r') as f:
        nUsers=int(f.readline().split(' ')[0])
    foldSize=int(math.ceil(nUsers/float(k)))
    ## insert -1 users to round the k folds and randomize the folds with less users
    usersList=range(nUsers)+[-1 for i in range(foldSize*k-nUsers)]
    ## shuffle users to generate k-folds
    random.shuffle(usersList)
    ## divide the users in folds
    usersFolds=[[user for user in usersList[i*foldSize:(i+1)*foldSize] if user>= 0] for i in range(k)]
    ratingFolds=[[rating.tolist() for rating in ratings if rating[0] in uFold] for uFold in usersFolds]
    foldLen=[len(foldd) for foldd in usersFolds]
    for i in range(k):
        mapAdd=[(fold!=i)*sum([foldLen[fold2]*(fold2!=i) for fold2 in range(fold)]) for fold in range(k)]
        userMap=[(indexUser,mapUser+mapAdd[indexFold]) for indexFold,foldd in enumerate(usersFolds) for mapUser,indexUser in enumerate(foldd)]
        userMap=sorted(userMap,key=lambda x:x[0])

        with open(folder+'/fold'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump((np.array([[userMap[rating[0]][1]]+rating[1:] for ratingList in ratingFolds[:i]+ratingFolds[i+1:] for rating in ratingList]),np.array([[userMap[rating[0]][1]]+rating[1:] for rating in ratingFolds[i]])), handle)

def k_fold_gen2(folder,filename,finfo,k):
    s=2*k
    ratings=read_ratings(filename)
    ratings.shape
    with open(finfo, 'r') as f:
        nUsers=int(f.readline().split(' ')[0])
	nItems=int(f.readline().split(' ')[0])
    foldSize=int(math.ceil(nUsers/float(s)))
    ## insert -1 users to round the k folds and randomize the folds with less users
    usersList=range(nUsers)+[-1 for i in range(foldSize*s-nUsers)]
    ## shuffle users to generate k-folds
    random.shuffle(usersList)
    ## divide the users in folds
    usersFolds=[[user for user in usersList[i*foldSize:(i+1)*foldSize] if user>= 0] for i in range(s)]
    ratingFolds=[[rating.tolist() for rating in ratings if rating[0] in uFold] for uFold in usersFolds]
    foldLen=[len(foldd) for foldd in usersFolds]
    for i in range(k):
        mapAdd=[(fold!=i)*sum([foldLen[fold2]*(fold2!=i) for fold2 in range(fold)]) for fold in range(k)]
        mapAdd=np.array([(fold!=(2*i)%s and fold!=(2*i+1)%s and fold!=(2*i+2)%s)*sum([foldLen[fold2]*(fold2!=(2*i)%s and fold2!=(2*i+1)%s and fold2!=(2*i+2)%s) for fold2 in range(fold)]) for fold in range(s)])
        mapAdd+=np.array([(fold!=(2*i)%s and (fold==(2*i+1)%s or fold==(2*i+2)%s))*sum([foldLen[fold2]*(fold2!=(2*i)%s and (fold2==(2*i+1)%s or fold2==(2*i+2)%s)) for fold2 in range(fold)]) for fold in range(s)])
        userMap=[(indexUser,mapUser+mapAdd[indexFold]) for indexFold,foldd in enumerate(usersFolds) for mapUser,indexUser in enumerate(foldd)]
        userMap=sorted(userMap,key=lambda x:x[0])

        with open(folder+'/fold'+str(i)+'.pickle', 'wb') as handle:
            train=np.array([[userMap[rating[0]][1],rating[1],rating[2]] for ratingList in ratingFolds[:2*i]+ratingFolds[2*(i+1)+1:] for rating in ratingList])
	    trainU=max(train[:,0])+1
	    trainI=nItems
            valid=np.array([[userMap[rating[0]][1],rating[1],rating[2]] for rating in ratingFolds[2*i]])
	    validU=max(valid[:,0])+1
	    validI=nItems
            test=np.array([[userMap[rating[0]][1],rating[1],rating[2]] for ratingList in (ratingFolds+[ratingFolds[0]])[2*i+1:2*i+3] for rating in ratingList])
	    testU=max(test[:,0])+1
	    testI=nItems
            pickle.dump((train,trainU,trainI,valid,validU,validI,test,testU,testI), handle)
    

def fold_load(folder,fold):
    with open(folder+'/fold'+str(fold)+'.pickle', 'rb') as handle:
        train,trainU,trainI,valid,validU,validI,test,testU,testI=pickle.load(handle)
    return train,trainU,trainI,valid,validU,validI,test,testU,testI

if __name__=='__main__':
    #print read_ratings('ml-100k/u.data')
    k_fold_gen2('ml-100k','ml-100k/u.data','ml-100k/u.info',5)
    #train,trainU,trainI,valid,validU,validI,test,testU,testI=fold_load('ml-100k',1)
    #print trainU+validU+testU
    #print max(train[:,0]),max(valid[:,0]),max(test[:,0])
