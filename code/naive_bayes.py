import numpy as np

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape
        C = 2
        
        # Compute mean_{d,c} and std_{d,c}
        # 1. split the data wrt. class
        X0, X1 = X[y==0], X[y==1]
        
        # 2. calculate means and std
        means = np.zeros((C, D))
        std = np.zeros((C,D))
        for d in range(D):
            means[0][d] = np.sum(X0[:,d]) / X0.shape[0]
            means[1][d] = np.sum(X1[:,d]) / X1.shape[0]
            
            std[0][d] = np.sum((means[0][d] - X0[:,d])**2) / X0.shape[0]
            std[1][d] = np.sum((means[1][d] - X1[:,d])**2) / X1.shape[0]  
        
        self.means, self.std = means, std 
        
        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N
        
        self.p_y = p_y      

    def predict(self, X):
        N, D = X.shape

        y_pred = np.zeros(N)
        
        for n in range(N):
            Xi = X[n, :]
            
            sum_pred0 = 0
            sum_pred1 = 0
            for d in range(D):
                Xi_d = Xi[d]
                
                mean0_d = self.means[0][d]
                std0_d = self.std[0][d]
                
                mean1_d = self.means[1][d]
                std1_d = self.std[1][d]
                
                pred0 = 0.5 * ((Xi_d-mean0_d) / np.sqrt(std0_d))**2 + np.log(np.sqrt(std0_d) * np.sqrt(2 * np.pi))
                pred1 = 0.5 * ((Xi_d-mean1_d) / np.sqrt(std1_d))**2 + np.log(np.sqrt(std1_d) * np.sqrt(2 * np.pi))

                sum_pred0 += pred0
                sum_pred1 += pred1
            
            
            prediction0 = -sum_pred0 + np.log(self.p_y[0])
            prediction1 = -sum_pred1 + np.log(self.p_y[1])
            
            y_pred[n] = 0 if prediction0 > prediction1 else 1
            
        
        return y_pred
