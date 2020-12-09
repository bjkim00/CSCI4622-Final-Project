import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from DataHandler import DataHandler
from sklearn.preprocessing import StandardScaler


class linModel():
    def __init__(self, pos = None, alpha = 1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = Lasso(alpha = self.alpha, max_iter=self.max_iter)
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        X_train = self.X_scaler.fit_transform(X_train)
        y_train = self.y_scaler.fit_transform(y_train)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        X = self.X_scaler.transform(X)
        return self.y_scaler.inverse_transform(self.model.predict(X))
    
    def predict_player(self, info_test, X_test, player_name, pos = None, ret_info = True):
        if pos is None:
            info_indices = info_test[info_test['Player'] == player_name].sort_index(ascending=True).index
        else:
            info_indices = info_test[pos][info_test[pos]['Player'] == player_name].sort_index(ascending=True).index
        X_test = X_test.loc[info_indices]
        X_test_normal = self.X_scaler.transform(X_test)
        
        if ret_info:
            return info_test.loc[info_indices], self.y_scaler.inverse_transform(self.model.predict(X_test_normal)), info_indices
        else:
            return self.y_scaler.inverse_transform(self.model.predict(X_test_normal)), info_indices
    
    def eval_accuracy(self, y, y_hat, scoring = None):
        if scoring is None:
            return r2_score(y, y_hat)
        else:
            return np.dot(r2_score(y, y_hat, multioutput = 'raw_values'), np.asarray(scoring))
    
    def cv_error(self, X, y, scoring = None):
        if scoring is None:
            return np.sqrt(-cross_val_score(self.model, X, y, cv = KFold(shuffle=True), scoring = 'neg_mean_squared_error'))
        else:
            return cross_val_score(self.model, X, y, cv = KFold(shuffle = True), scoring = lambda estimator, X, y: np.dot(np.sqrt(mean_squared_error(y, estimator.predict(X), multioutput = 'raw_values')), np.asarray(scoring)))

    def eval_error(self, y, y_hat, scoring = None):
        if scoring is None:
            return np.sqrt(mean_squared_error(y, y_hat))
        else:
            return np.dot(np.sqrt(mean_squared_error(y, y_hat, multioutput = 'raw_values')), np.asarray(scoring))