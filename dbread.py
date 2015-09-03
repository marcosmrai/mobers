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
    ratings = np.array([[u-1,i-1,r] for u,i,r in ratings])
  return ratings

def read_user_ratings(filename, user_id):
  with open(filename, 'r') as f:
    user_ratings = [(int(line.split()[1]), int(line.split()[2]))\
      for line in f if int(line.split()[0])==(user_id+1)]
  return user_ratings

def twoxk(folder,filename,finfo,k,rate=0.1):
  ratings=read_ratings(filename)
  # read the infos about the data
  with open(finfo, 'r') as f:
    nUsers=int(f.readline().split(' ')[0])
    nItems=int(f.readline().split(' ')[0])

  # generate and shuffle the list of users
  users = range(nUsers)
  random.shuffle(users)

  # separate rate*100 % of the users to test and put the excedent in vtUsers
  testUsers = users[:int(math.ceil(nUsers*rate))]
  vtUsers = users[int(math.ceil(nUsers*rate)):]
  nvtUsers = len(vtUsers)


  for i in range(k):
    # shuffle the data for every 2fold generation
    random.shuffle(vtUsers)

    for invert in range(2):
      # invert the list to the first half be the second half now
      vtUsers.reverse()

      # the fisrt half is the train, and the second is the validation
      trainUsers = vtUsers[:nvtUsers/2]
      valUsers = vtUsers[nvtUsers/2:]

      with open(folder+'/fold'+str(2*i+invert)+'.pickle', 'wb') as handle:
        # all the ratings which belong to trainUsers list will be in the train
        train=np.array([[trainUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in trainUsers])
        trainU=max(train[:,0])+1
        trainI=nItems

        # all the ratings which belong to valUsers list will be in the validadion
        valid=np.array([[valUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in valUsers])
        validU=max(valid[:,0])+1
        validI=nItems

        # all the ratings which belong to testUsers list will be in the test
        test=np.array([[testUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in testUsers])
        testU=max(test[:,0])+1
        testI=nItems

        #save the data
        pickle.dump((train,trainU,trainI,valid,validU,validI,test,testU,testI), handle)

        print 'Sort',i,'Fold',invert
        print 'Card Training (u/i/r)', trainU, trainI, len(train)
        print 'Card Validation (u/i/r)', validU, validI, len(valid)
        print 'Card Testing (u/i/r)', testU, testI, len(test)
        print '\n\n'

def kfold(folder,filename,finfo,k,rate=0.1):
  ratings=read_ratings(filename)
  # read the infos about the data
  with open(finfo, 'r') as f:
    nUsers=int(f.readline().split(' ')[0])
    nItems=int(f.readline().split(' ')[0])

    # generate and shuffle the list of users
    users = range(nUsers)
    random.shuffle(users)

    # separate rate*100 % of the users to test and put the excedent in vtUsers
    testUsers = users[:int(math.ceil(nUsers*rate))]
    vtUsers = users[int(math.ceil(nUsers*rate)):]
    nvtUsers = len(vtUsers)

    for i in range(k):
      # card fold determine the cardinality of the each fold
      cardFold = [nvtUsers/k+int(j<(nvtUsers%k)) for j in range(k)]

      # determine the folds
      trainUsers = vtUsers[:sum(cardFold[:i])]+vtUsers[sum(cardFold[:i+1]):]
      valUsers = vtUsers[sum(cardFold[:i]):sum(cardFold[:i+1])]

      with open(folder+'/fold'+str(i)+'.pickle', 'wb') as handle:
        # all the ratings which belong to trainUsers list will be in the train
        train=np.array([[trainUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in trainUsers])
        trainU=max(train[:,0])+1
        trainI=nItems

        # all the ratings which belong to valUsers list will be in the validadion
        valid=np.array([[valUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in valUsers])
        validU=max(valid[:,0])+1
        validI=nItems

        # all the ratings which belong to testUsers list will be in the test
        test=np.array([[testUsers.index(rating[0]),rating[1],rating[2]] for rating in ratings if rating[0] in testUsers])
        testU=max(test[:,0])+1
        testI=nItems

        print 'Fold',i
        print 'Card Training (u/i/r)', trainU, trainI, len(train)
        print 'Card Validation (u/i/r)', validU, validI, len(valid)
        print 'Card Testing (u/i/r)', testU, testI, len(test)
        print '\n\n'

        #save the data
        pickle.dump((train,trainU,trainI,valid,validU,validI,test,testU,testI), handle)

def freqItems(folder,filename,finfo):
  ratings=read_ratings(filename)
  # read the infos about the data
  with open(finfo, 'r') as f:
    nUsers=int(f.readline().split(' ')[0])
    nItems=int(f.readline().split(' ')[0])

    nEachItem = np.zeros(nItems)
    for rating in ratings:
        nEachItem[rating[1]]+=1
    relevantItems = np.argsort(nEachItem).tolist()
    nEachItem = np.sort(nEachItem).tolist()
    itemFreq = [dupla for dupla in zip(relevantItems,nEachItem)]
    with open(folder+'/itemfreq.pickle', 'wb') as handle:
        pickle.dump(itemFreq, handle)

def loadFreqItems(folder):
  with open(folder+'/itemfreq.pickle', 'rb') as handle:
     itemFreq=pickle.load(handle)
  return itemFreq


def fold_load(folder,fold):
  with open(folder+'/fold'+str(fold)+'.pickle', 'rb') as handle:
    train,trainU,trainI,valid,validU,validI,test,testU,testI=pickle.load(handle)
  return train,trainU,trainI,valid,validU,validI,test,testU,testI

def datagen(ifolder):
  kfold(ifolder,ifolder+'/u.data',ifolder+'/u.info',5)

if __name__=='__main__':
  #kfold('ml-100k','ml-100k/u.data','ml-100k/u.info',5)
  freqItems('ml-100k','ml-100k/u.data','ml-100k/u.info')
  print loadFreqItems('ml-100k')
