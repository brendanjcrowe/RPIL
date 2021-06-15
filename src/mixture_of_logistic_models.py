import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import math

class MixtureOfLogits(object):
    
    def __init__(self, C=0, alpha=np.exp(-5), min_iter=100, max_iter=1000, criterion=0.1):
        
        self.C = C
        self.alpha = alpha
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.start = datetime.now()
        self.end = datetime.now()
        
    def fit(self, X, y, demos, M, n_models=2):
        
        self.encoder = LabelBinarizer().fit(y)
        self.Y = self.encoder.transform(y)
        self.y = y
        self.M = M
        self.D = demos
        self.N = len(self.D)
        self.n_models = n_models
        self.n_classes = self.Y.shape[1]
        self.X = X
        self.n_examples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.iteration = 0
        self.gammas = []
        self.initialize_models()
        
        while not self.mixture_convergence():
            
            self.expectation()
            self.maximization()
            self.gammas.append([self.models[0][1],self.models[1][1]])
            self.iteration += 1
        
        return self
            
    def initialize_models(self):
        
        self.models = []
        
        for i in range(self.n_models):
            
            self.models.append(
                [
                    1.0/self.n_models,
                    np.zeros(self.X.shape[0]),
                    LogisticRegression(
                        penalty='none',
                        multi_class='multinomial',
                        warm_start=True,
                        fit_intercept=True,
                        n_jobs=-1,
                        max_iter=i*1000
                    ).fit(self.X, self.y),
                    np.random.normal(
                        0,
                        np.std(self.X),
                        (self.n_classes, self.n_features)
                    )
                ]
            )
            
#             for j in []
#             model[2].coef_ = np.random.normal
            
            
    def expectation(self):
        

        Z = np.zeros(self.X.shape[0])
        for model in self.models:

            pi = model[0]
            gamma = np.zeros(self.X.shape[0])
            probabilities = self.probabilities(model[2])

            for d in self.D:

                gamma[d[0]:d[1]+1] = pi * np.product(probabilities[d[0]:d[1]+1])

            model[1] = gamma
            Z += gamma

        for model in self.models:

            model[1] /= Z
            
    def maximization(self):
        
        for i, model in enumerate(self.models):
            
            phi = (self.N - self.M) / self.N if i == 1 else self.M / self.N
            pi = model[0]
            gamma = model[1]
            W = model[3]
            Model = model[2]
            Model.fit(self.X, self.y, sample_weight=gamma)
            probs = self.probabilities(Model)
            model[0] = phi #* (np.sum(np.array([np.product(probs[d[0]:d[1]+1]) for d in self.D])) / len(self.D))
            model[2] = Model
            model[3] = np.column_stack((Model.coef_, Model.intercept_))
                
#     def gradient(w, XX, y, pi, gamma):
        
#         X = np.einsum('ijk, j -> ijk', XX, gamma)
#         return np.matmul(X.T, self.Y) - np.matmul(np.exp(np.matmul(X, W)).T /np.sum(np.exp(np.matmul(X, W)), axis=1), X).T + C * w
# #         return   ( np.multiply(np.matmul(self.X.T, self.Y) - np.matmul(
# #             np.exp(
# #                 np.matmul(self.X, W)
# #             ).T / np.sum(np.exp(np.matmul(self.X, W)), axis=1), self.X
# #         ), gamma).T + C * W)
    
    
#     def gradient_convergence(self, W, i):
        
#         return i > 100 and i < 1000
    
    def mixture_convergence(self):
        
        return self.iteration > 100

    def probabilities(self, model):
        
        probs = model.predict_proba(self.X)
        return  np.amax(np.multiply(probs, self.Y), axis=1)
#        return np.product(model.predict_proba(self.X), axis=1)
    
    def neg_log_likelihood():
        
        return None
        
    def p_X(self, W):

        px = np.matmul(self.X, W)
        return (px.T / np.sum(px, axis=1)).T
    
    def predict(self, X):
        
        try:
            for model in enumerate(self.models):
                xs = np.exp(np.matmul(X, model[2]))
                preds = (xs.T / np.sum(xs, axis=1)).T
                b = np.zeros(preds.shape)
                b[np.arange(len(preds)), preds.argmax(1)] = 1
                yield self.Encoder.inverse_transform(b)
        except Exception as exe:
            print('No model trained yet')
            print(exe)
            return None
        
        
    def result(self):
        
        return self.models[0][2], self.models[1][2]
    
class Expert_(object):
        
        def __init__(self,  X, y, n_classes, n_features, pi):
            
            self.pi = pi
            self.gamma = np.zeros(X.shape[0])
            self.model = LogisticRegression(
                        penalty='none',
                        multi_class='multinomial',
                        warm_start=True,
                        fit_intercept=True,
                        n_jobs=-1,
                        max_iter=10000
            ).fit(X, y)
            self.weights = np.random.normal(
                0,
                np.std(X),
                (n_classes, n_features)
            )
            #self.model.coef_ = self.weights
            #self.model.intercept_ = np.random.normal(0, np.std(X), (n_classes,))
            
class Adversary_(object):

    def __init__(self, X, pi):

        self.gamma = np.zeros(X.shape[0])
        self.pi = pi
        
class AdversarialLogits(object):
    
    def __init__(self, C=0, min_iter=100, max_iter=1000, verbose=False):
        
        self.C = C
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.start = datetime.now()
        self.end = datetime.now()
        self.verbose = verbose
        
        
    def fit(self, X, y, demos, pi):
        
        self.encoder = LabelBinarizer().fit(y)
        self.Y = self.encoder.transform(y)
        self.y = y
        self.D = demos
        self.n_classes = self.Y.shape[1]
        self.X = X
        self.pi = pi
        self.N = len(demos)
        self.n_examples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.iteration = 0
        self.initialize_models()
        
        while not self.mixture_convergence():
            
            self.expectation()
            self.maximization()
            self.iteration += 1
        
        return self
    
    
            
    def initialize_models(self):
        
        self.expert = Expert_(self.X, self.y, self.n_classes, self.n_features, self.pi)
        self.adversary = Adversary_(self.X, 1-self.pi)
        

            
    def expectation(self):
        
        probs_expert = self.probabilities(self.expert.model)
        probs_adversary = (1 - probs_expert) #* self.adversary.pi
        #probs_expert *= self.expert.pi
        for d in self.D:
            
            gamma_expert = np.nan_to_num(np.product(probs_expert[d[0]:d[1]+1]) * self.expert.pi)
            #if math.isnan(gamma_expert): gamma_expert = 0.
            gamma_adversary = np.nan_to_num(np.product(probs_adversary[d[0]:d[1]+1]) * self.adversary.pi)
            #if math.isnan(gamma_adversary): gamma_adversary = 0.
            self.expert.gamma[d[0]:d[1]+1] = gamma_expert
            self.adversary.gamma[d[0]:d[1]+1] = gamma_adversary

        Z = self.expert.gamma + self.adversary.gamma
        self.expert.gamma /= Z
        self.adversary.gamma /= Z
        self.expert.gamma = np.nan_to_num(self.expert.gamma)
        self.adversary.gamma = np.nan_to_num(self.adversary.gamma)
        if math.isnan(gamma_adversary): gamma_adversary = 0.
            
        if self.verbose:
            print(*['Expert gamma for {}: {}\n'.format(i, np.unique(self.expert.gamma[d[0]:d[1]+1])) for i, d in enumerate(self.D)])
            print(*['Adversary gamma for {}: {}\n'.format(i, np.unique(self.adversary.gamma[d[0]:d[1]+1])) for i, d in enumerate(self.D)])
            #self.adversary.gamma[d[0]:d[1]+1] = self.adversary.pi * 1
#         self.expert.gamma *= self.expert.pi
#         self.adversary.gamma[:] = (self.N-self.M)/(probs2 * self.N)
        
            
    def maximization(self):
        
        self.expert.model.fit(self.X, self.y, sample_weight=self.expert.gamma)
        self.expert.weights = np.column_stack((self.expert.model.coef_, self.expert.model.intercept_))
                
    def mixture_convergence(self):
        
        return self.iteration > self.min_iter and self.iteration < self.max_iter

    def probabilities(self, model):
        
        probs = model.predict_proba(self.X)
        return  np.amax(np.multiply(probs, self.Y), axis=1)
        
    def p_X(self, W):

        px = np.matmul(self.X, W)
        return (px.T / np.sum(px, axis=1)).T
    
    def predict(self, X):
        
        try:
            for model in enumerate(self.models):
                xs = np.exp(np.matmul(X, model[2]))
                preds = (xs.T / np.sum(xs, axis=1)).T
                b = np.zeros(preds.shape)
                b[np.arange(len(preds)), preds.argmax(1)] = 1
                yield self.Encoder.inverse_transform(b)
        except Exception as exe:
            print('No model trained yet')
            print(exe)
            return None
        
        
    def result(self):
        
        return self.expert, self.adversary
