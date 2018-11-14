class MatrixFactorization:
    
    #initializing the user-movie rating matrix, k: number of latent features, alpha, beta
    def __init__(self, R, K, alpha, beta, max_iter):
        self.R = R
        self.K = K
        self.num_users, self.num_movies = R.shape
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
    
    #Initializing P: user-feature and Q: movie-feature matrix
    def train(self):
        self.P = np.random.normal(scale = 1./self.K, size = (self.num_users, self.K))
        self.Q = np.random.normal(scale = 1./self.K, size = (self.num_movies, self.K))
    
        #Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_movies)
        self.b = np.mean(self.R[np.where(self.R != 0)]) # mean ratings
        
        # List of training examples
        self.samples = [(i,j,self.R[i, j]) for i in range(self.num_users) for j in range(self.num_movies) if self.R[i, j] > 0]
        
        #SGD: Stochastic Gradient Descent for given number of iterations
        training_process = []
        for i in range(self.max_iter):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i,mse))
            if (i+1) % 10 == 0:
                print("Iteration: " + str(i+1) + " ; error: " + str(mse))
            
        return training_process
    
    #Computing total mse: mean squared error
    def mse(self):
        xs, ys = self.R.nonzero() # returns indices of non-zero values/ratings
        predicted = self.full_matrix()
        error = 0
        for x,y in zip(xs, ys):
            error += pow((self.R[x, y] - predicted[x, y]), 2)
        return np.sqrt(np.mean(error))
    
    #Computing SGD
    def sgd(self):
        for i,j,r in self.samples:
            prediction = self.get_rating(i,j)
            e = r - prediction
            
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            self.P[i,:] += self.alpha * (e * self.Q[j,:] - self.beta * self.P[i,:])
            self.Q[j,:] += self.alpha * (e * self.P[i,:] - self.beta * self.Q[j,:])
           
    
    #Ratings for user i and movie j
    def get_rating(self, i, j):
        pred = self.b + self.b_u[i] + self.b_i[j] + self.P[i,:].dot(self.Q[j,:].T)
        return pred
        
    #full user-movie rating matix
    def full_matrix(self):
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis,:] + mf.P.dot(mf.Q.T)

