import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer

class LogisticRegression(object):
    
    def __init__(self, C=0, alpha=np.exp(-5), w_start='', min_iter=100, max_iter=1000, criterion=0.1, bold_driver=False, diagnostics=True):
        
        self._C = C
        self._alpha = alpha
        self._w_start = w_start
        self._min_iter = min_iter
        self._max_iter = max_iter
        self._iterations = []
        self._neg_log_likes = []
        self._misclassification_rates = []
        self._tp = []
        self._fp = []
        self._tn = []
        self._fn = []
        self._gradients = []
        self._delta_grads = []
        self._criterion = criterion
        self._diagnostics = diagnostics
        self._bold_driver = bold_driver
        self._ws = []
        self._start = datetime.now()
        self._end = datetime.now()
        

    def fit(self, X, y, mini_batch=False):
        
        self._start = datetime.now()
        stop = False
        iteration = 0
        
        if self._w_start == '':
            self._w = np.random.normal(0, np.std(X), X.shape[1])
        else:
            self._w = self._w_start
        
        self._ws.append(np.copy(self._w))
        self._neg_log_likes.append(self.neg_log_like(X, y, C=self._C))
        
        if self._diagnostics:
            self._iterations.append(iteration)
            self._misclassification_rates.append(self.misclassification_rate(X, y))
            confuse = self.confusion_matrix(X, y)
            self._tp.append(confuse[0])
            self._fp.append(confuse[1])
            self._tn.append(confuse[2])
            self._fn.append(confuse[3])
            self._gradients.append(self.gradient(X, y, C=self._C))
            self._delta_grads.append(0)
        
        while not stop:
            
            if mini_batch:
                indicies = np.random.choice(X.shape[0], size=mini_batch, replace=False)
                X_batch = X[indicies, :]
                y_batch = y[indicies]
            else:
                X_batch = X
                y_batch = y
                
            grad = self.gradient(X_batch, y_batch, C=self._C)
            self._w = self._w - self._alpha * grad
            iteration += 1
            self._ws.append(np.copy(self._w))
            self._neg_log_likes.append(self.neg_log_like(X, y, C=self._C))
            
            if self._diagnostics:
                self._iterations.append(iteration)
                self._misclassification_rates.append(self.misclassification_rate(X, y))
                confuse = self.confusion_matrix(X, y)
                self._tp.append(confuse[0])
                self._fp.append(confuse[1])
                self._tn.append(confuse[2])
                self._fn.append(confuse[3])
                self._gradients.append(grad)
                self._delta_grads.append(self._gradients[-1] - self._gradients[-2])

            self._alpha = self.new_alpha()
            stop =  (iteration > self._min_iter and
                        (iteration > self._max_iter or
                        self.converged(X, y)))
            
        self._w = self._ws[self._neg_log_likes.index(min(self._neg_log_likes))]   
        self._end = datetime.now()
        return self
        
    def new_alpha(self):
        
        if not self._bold_driver:
            return self._alpha
        else:
            if self._neg_log_likes[-2] <= self._neg_log_likes[-1]:
                return self._alpha * self._bold_driver[1]
            else:
                return self._alpha * self._bold_driver[0]
        
    def converged(self, X, y):
        
        return np.max(np.absolute(self.gradient(X, y, self._C))) < self._criterion or np.mean(self._neg_log_likes[-int(self._min_iter/2):]) <= self._neg_log_likes[-1]
    
    def gradient(self, X, y, C=0):
        return np.matmul((y - ( 1 / (1 + np.exp(np.matmul(X, self._w))))).T, X) + C * self._w
    
    def neg_log_like(self, X, y, C=0):
        mu = np.matmul(X, self._w)
        return - np.sum((1-y) * mu - self.log1PlusExp(mu)) + C/2 * np.dot(self._w, self._w)
    
    def log1PlusExp(self, mu):
        m = np.max(mu)
        return m + np.log(np.exp(-m) + np.exp(mu - m))
    
    def misclassification_rate(self,X,y):
        """
        w: array of shape (nFeatures,). Weights.
        x: array of shape (nExamples, nFeatures). Observed features.
        y: array of shape (nExamples,). Observed binary labels (either 0 or 1).
        """
        return np.sum(np.fromiter((self.predict(X) != y),int)) / len(y)

    def confusion_matrix(self, X,y):
        """
        Returns (TP, FP, TN, FN)
        """
        y_hat = self.predict(X)
        return np.count_nonzero(np.logical_and(y_hat == 1, y == 1)), \
               np.count_nonzero(np.logical_and(y_hat == 1, y == 0)), \
               np.count_nonzero(np.logical_and(y_hat == 0, y == 0)), \
               np.count_nonzero(np.logical_and(y_hat == 0, y == 1))

    def predict(self, X):
        
        try:
            return np.array([1 if 1 > y else 0 for y in np.exp(np.matmul(X, self._w))])
        except Exception as exe:
            print('No model trained yet')
            print(exe)
            return None
        
    def parameters(self):
        try:
            return self._w
        except Exception as exe:
            print(exe)
            print('No model has been fit yet.')
            return None
        
    def training_time(self):
        return self._end - self._start
        
    def diagnostics(self):
            
        return pd.DataFrame(
            data = np.array(
                [
                    self._neg_log_likes,
                    self._misclassification_rates,
                    self._tp,
                    self._fp,
                    self._tn,
                    self._fn,
                    self._gradients,
                    self._delta_grads
                ], dtype=object
            ).T,
            index = self._iterations,
            columns = [
                'neg_log_like',
                'misclassification_rates',
                'true_positives',
                'false_positives',
                'true_negatives',
                'false_negatives',
                'gradients',
                'delta_gradients'
            ]     
        )
    
    


class MultinomialLogisticRegression(object):
    
    def __init__(self, C=0, alpha=np.exp(-5), w_start='', min_iter=100, max_iter=1000, criterion=0.1, bold_driver=False, diagnostics=True):
        
        self._C = C
        self._alpha = alpha
        self._w_start = w_start
        self._min_iter = min_iter
        self._max_iter = max_iter
        self._iterations = []
        self._neg_log_likes = []
        self._misclassification_rates = []
        self._gradients = []
        self._delta_grads = []
        self._criterion = criterion
        self._diagnostics = diagnostics
        self._bold_driver = bold_driver
        self._ws = []
        self._start = datetime.now()
        self._end = datetime.now()
        

    def fit(self, X, y, mini_batch=False):
        
        self._start = datetime.now()
        stop = False
        iteration = 0
        self.Encoder = LabelBinarizer().fit(y)
        classes = self.Encoder.classes_
        n_classes = classes.shape[0]
        n_features = X.shape[1]
        n_examples = X.shape[0]
        y = self.Encoder.transform(y)
        
        if self._w_start == '':
            self._w = np.random.normal(0, np.std(X), (n_features, n_classes))
        else:
            self._w = self._w_start
        
        self._ws.append(np.copy(self._w))
        self._neg_log_likes.append(self.neg_log_like(X, y, C=self._C))
        
        if self._diagnostics:
            self._iterations.append(iteration)
            self._misclassification_rates.append(self.misclassification_rate(X, y))
            self._gradients.append(self.gradient(X, y, C=self._C))
            self._delta_grads.append(0)
        
        while not stop:
            
            if mini_batch:
                indicies = np.random.choice(X.shape[0], size=mini_batch, replace=False)
                X_batch = X[indicies, :]
                y_batch = y[indicies]
            else:
                X_batch = X
                y_batch = y
                
            grad = self.gradient(X_batch, y_batch, C=self._C)
            self._w += self._alpha * grad
            iteration += 1
            self._ws.append(np.copy(self._w))
            self._neg_log_likes.append(self.neg_log_like(X, y, C=self._C))
            
            if self._diagnostics:
                self._iterations.append(iteration)
                self._misclassification_rates.append(self.misclassification_rate(X, y))
                self._gradients.append(grad)
                self._delta_grads.append(self._gradients[-1] - self._gradients[-2])

            self._alpha = self.new_alpha()
            stop =  (iteration > self._min_iter and
                        (iteration > self._max_iter or
                        self.converged(X, y)))
            
        self._w = self._ws[self._neg_log_likes.index(min(self._neg_log_likes))]   
        self._end = datetime.now()
        return self
        
    def new_alpha(self):
        
        if not self._bold_driver:
            return self._alpha
        else:
            if self._neg_log_likes[-2] >= self._neg_log_likes[-1]:
                return self._alpha * self._bold_driver[1]
            else:
                return self._alpha * self._bold_driver[0]
        
    def converged(self, X, y):
        
        return np.max(np.absolute(self.gradient(X, y, self._C))) < self._criterion or np.mean(self._neg_log_likes[-int(self._min_iter/2):]) <= self._neg_log_likes[-1]
    
    def gradient(self, X, y, C=0):
        
        return np.matmul(X.T, y) - np.matmul(np.exp(np.matmul(X, self._w)).T /np.sum(np.exp(np.matmul(X, self._w)), axis=1), X).T + C * self._w
        #return  np.matmul(y.T, X) - np.exp(np.matmul(np.matmul(y, self.w).T, X)) / np.sum(np.matmul(self.w, X.T)) + C
        #return np.matmul((y - ( 1 / (1 + np.exp(np.matmul(X, self._w))))).T, X) + C * self._w
    
    def neg_log_like(self, X, y, C=0):
        
        return np.sum(np.matmul(y, (np.matmul(X, self._w).T - np.log(np.sum(np.exp(np.matmul(X, self._w)), axis=1)))))
    
    def log1PlusExp(self, mu):
        m = np.max(mu)
        return m + np.log(np.exp(-m) + np.exp(mu - m))
    
    def misclassification_rate(self,X,y):
        """
        w: array of shape (nFeatures,). Weights.
        x: array of shape (nExamples, nFeatures). Observed features.
        y: array of shape (nExamples,). Observed binary labels (either 0 or 1).
        """
        return np.sum(self.predict(X) != self.Encoder.inverse_transform(y)) / y.shape[0]

    def confusion_matrix(self, X,y):
        """
        Returns (TP, FP, TN, FN)
        """
        y_hat = self.predict(X)
        return np.count_nonzero(np.logical_and(y_hat == 1, y == 1)), \
               np.count_nonzero(np.logical_and(y_hat == 1, y == 0)), \
               np.count_nonzero(np.logical_and(y_hat == 0, y == 0)), \
               np.count_nonzero(np.logical_and(y_hat == 0, y == 1))

    def predict(self, X):
        
        try:
            xs = np.exp(np.matmul(X, self._w))
            preds = (xs.T / np.sum(xs, axis=1)).T
            b = np.zeros(preds.shape)
            b[np.arange(len(preds)), preds.argmax(1)] = 1
            return self.Encoder.inverse_transform(b)
        except Exception as exe:
            print('No model trained yet')
            print(exe)
            return None
        
    def parameters(self):
        try:
            return self._w
        except Exception as exe:
            print(exe)
            print('No model has been fit yet.')
            return None
        
    def training_time(self):
        return self._end - self._start
        
    def diagnostics(self):
            
        return pd.DataFrame(
            data = np.array(
                [
                    self._neg_log_likes,
                    self._misclassification_rates,
                    self._gradients,
                    self._delta_grads
                ], dtype=object
            ).T,
            index = self._iterations,
            columns = [
                'neg_log_like',
                'misclassification_rates',
                'gradients',
                'delta_gradients'
            ]     
        )