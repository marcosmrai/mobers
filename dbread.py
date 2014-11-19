import numpy as np

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
        ratings = [(u-1,i-1,r) for u,i,r in ratings]
    return ratings
    
def read_user_ratings(filename, user_id):
    with open(filename, 'r') as f:
        user_ratings = [(int(line.split()[1]), int(line.split()[2]))\
                       for line in f if int(line.split()[0])==user_id]
    return user_ratings
            
if __name__=='__main__':
    print read_ratings('ml-100k/u.data')
