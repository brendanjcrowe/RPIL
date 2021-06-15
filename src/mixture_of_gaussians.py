import numpy as np
import scipy.stats as st
from sklearn.cluster import KMeans

class MixtureOfGaussians(object):
    
    
    def __init__(self, min_iter=100, max_iter=1000, crit=0.01):
        
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.crit = crit
        self.log_like = []
        
        return None
        
        
    def fit(self, X, n_gaussians):
        
        self.X = X
        self.Z = 0
        self.iteration = 0
        self.initialize_gaussians(n_gaussians)
        
        while not self.converged():
            self.E_Step()
            self.M_Step()
            self.log_likelihood()
            self.iteration += 1
            
        return self
            
        
    def initialize_gaussians(self, n_gaussians):

        self.gaussians = []

        kmeans = KMeans(n_gaussians).fit(self.X)
        mu_k = kmeans.cluster_centers_

        for mean in mu_k:
            self.gaussians.append(
                {
                    'pi_k': 1.0 / n_gaussians,
                    'mu_k': mean,
                    'cov_k': np.identity(self.X.shape[1], dtype=np.float64)
                }
            )
        
        
    
        
    def E_Step(self):
        
        self.Z = np.zeros(self.X.shape[0])

        for gaussian in self.gaussians:
            pi_k = gaussian['pi_k']
            mu_k = gaussian['mu_k']
            cov_k = gaussian['cov_k']

            gamma_k = pi_k * st.multivariate_normal.pdf(x=self.X, mean=mu_k, cov=cov_k)
            self.Z += gamma_k

            gaussian['gamma_k'] = gamma_k

        for gaussian in self.gaussians:
            gaussian['gamma_k'] /= self.Z
            
            
            
    def M_Step(self):
        
        N = float(self.X.shape[0])

        for gaussian in self.gaussians:
            
            gamma_k = gaussian['gamma_k']
            #cov_k = np.zeros((self.X.shape[1], self.X.shape[1]))
            N_k = np.sum(gamma_k, axis=0)
            pi_k = N_k / N
            mu_k = np.matmul(gamma_k.T, self.X) / N_k
            err = ((self.X - mu_k).T * np.sqrt(gamma_k)).T
            cov_k = np.matmul(err.T, err) / N_k
#             for j in range(self.X.shape[0]):
#                 diff = self.X[j] - mu_k
#                 cov_k += gamma_k[j] * np.outer(diff, diff.T)

#             cov_k /= N_k

            gaussian['pi_k'] = pi_k
            gaussian['mu_k'] = mu_k
            gaussian['cov_k'] = cov_k
            
            
    def log_likelihood(self):
        
        self.log_like.append(np.sum(np.log(self.Z)))
        
    
    def converged(self):
        
        return (self.iteration > self.min_iter and self.log_like[-10] / self.log_like[-1] < self.crit) or self.iteration > self.max_iter
        
    
    def result(self):
        
        return self.gaussians, self.log_like
    
    
    
    
class MixtureOfGaussianDemonstrators(object):
    
    def __init__(self, min_iter=100, max_iter=1000, crit=0.01):
        
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.crit = crit
        self.log_like = []
        
        return None
        
    def fit(self, X, n_gaussians, demos):
        
        self.X = X
        self.Z = 0
        self.D = demos
        self.iteration = 0
        self.initialize_gaussians(n_gaussians)
        
        while not self.converged():
            self.E_Step()
            self.M_Step()
            self.log_likelihood()
            self.iteration += 1
            
        return self
            
        
        
    def initialize_gaussians(self, n_gaussians):

        self.gaussians = []

        kmeans = KMeans(n_gaussians).fit(self.X)
        mu_k = kmeans.cluster_centers_

        for mean in mu_k:
            self.gaussians.append(
                {
                    'pi_k': 1.0 / n_gaussians,
                    'mu_k': mean,
                    'cov_k': np.identity(self.X.shape[1], dtype=np.float64)
                }
            )
            
        
    def E_Step(self):
        
        self.Z = np.zeros(self.X.shape[0])

        for gaussian in self.gaussians:
            pi_k = gaussian['pi_k']
            mu_k = gaussian['mu_k']
            cov_k = gaussian['cov_k']
            gamma_k = np.zeros(self.X.shape[0])
            probs = st.multivariate_normal.pdf(x=self.X, mean=mu_k, cov=cov_k)
            
            for i in self.D:
                
                gamma_k[i[0]:i[1]+1] = pi_k * np.product(probs[i[0]:i[1]+1])
                
            self.Z += gamma_k

            gaussian['gamma_k'] = gamma_k

        for gaussian in self.gaussians:
            gaussian['gamma_k'] /= self.Z
            
            
            
    def M_Step(self):
        
        N = self.X.shape[0]

        for gaussian in self.gaussians:
            gamma_k = gaussian['gamma_k']
            cov_k = np.zeros((self.X.shape[1], self.X.shape[1]))

            N_k = np.sum(gamma_k, axis=0)

            pi_k = N_k / N
            mu_k = np.matmul(gamma_k.T, self.X) / N_k
            
            err = ((self.X - mu_k).T * np.sqrt(gamma_k)).T
            cov_k = np.matmul(err.T, err) / N_k
            
#             for j in range(self.X.shape[0]):
#                 diff = (self.X[j] - mu_k).reshape(-1, 1)
#                 cov_k += gamma_k[j] * np.dot(diff, diff.T)

#             cov_k /= N_k

            gaussian['pi_k'] = pi_k
            gaussian['mu_k'] = mu_k
            gaussian['cov_k'] = cov_k
            
            
    def log_likelihood(self):
        
        self.log_like.append(np.sum(np.log(self.Z)))
        
    
    def converged(self):
        
        return (self.iteration > self.min_iter and self.log_like[-10] / self.log_like[-1] < self.crit) or self.iteration > self.max_iter
        
    
    def results(self):
        
        return self.gaussians, self.log_like
    

    