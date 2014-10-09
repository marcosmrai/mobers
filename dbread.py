import numpy

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
        u.append(2 * numpy.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * numpy.random.randn(latent_dimension))
        
    # Get num_ratings ratings per user.
    for i in range(num_users):
        items_rated = numpy.random.permutation(num_items)[:num_ratings]

        for jj in range(num_ratings):
            j = items_rated[jj]
            rating = numpy.sum(u[i] * v[j]) + noise * numpy.random.randn()
        
            ratings.append((i, j, rating))  # thanks sunquiang

    return (ratings, u, v)

