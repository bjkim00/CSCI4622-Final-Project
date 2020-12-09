import numpy as np

from sklearn.linear_model import Lasso
from DataHandler import DataHandler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pickle import load



class predModel:
    def __init__(self, pos = None, year_range = [1999, 2019], offset = 16, batch_size = 32, epochs = 100):
        self.offset = offset
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.stat_num = 13
        self.model = Sequential()
        # self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model.add(LSTM(units = self.batch_size, input_shape = (self.offset, self.stat_num)))
        self.model.add(Dense(self.stat_num))
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')


    def reshape_data(self, X):
        X = np.asarray(X)
        X = np.reshape(X, (X.shape[0],-1, self.stat_num))
        return X


    def train(self, X_train, y_train, verbose = 1):
        X_train_normal = self.X_scaler.fit_transform(X_train)
        y_train_normal = self.y_scaler.fit_transform(y_train)
        X_train_normal = self.reshape_data(X_train_normal)
        
        self.model.fit(X_train_normal, y_train_normal, epochs = self.epochs, batch_size = self.batch_size, verbose = verbose)

    
    def predict_player(self, info_test, X_test, player_name, pos = None, ret_info = True):
        if pos is None:
            info_indices = info_test[info_test['Player'] == player_name].sort_index(ascending=True).index
        else:
            info_indices = info_test[pos][info_test[pos]['Player'] == player_name].sort_index(ascending=True).index
        X_test = X_test.loc[info_indices]
        X_test_normal = self.X_scaler.transform(X_test)
        X_test_normal = self.reshape_data(X_test_normal)
        
        if ret_info:
            return info_test.loc[info_indices], self.y_scaler.inverse_transform(self.model.predict(X_test_normal)), info_indices
        else:
            return self.y_scaler.inverse_transform(self.model.predict(X_test_normal)), info_indices
    
    def predict(self, X):
        X_normal = self.X_scaler.transform(X)
        X_normal = self.reshape_data(X_normal)
        return self.y_scaler.inverse_transform(self.model.predict(X_normal))


    def eval_accuracy(self, y, y_hat, scoring = None):
        if scoring is None:
            return r2_score(y, y_hat)
        else:
            return np.dot(r2_score(y, y_hat, multioutput = 'raw_values'), np.asarray(scoring))
    

    def cv_error(self, X, y, scoring = None):
        cv = KFold(shuffle = True, n_splits = 5)
        errs = np.zeros(13)
        X = np.asarray(X)
        y = np.asarray(y)
        for train, test in cv.split(X, y):
            self.train(X[train], y[train], verbose = 0)
            y_pred = self.predict(X[test])
            thing = mean_squared_error(y[test], y_pred, multioutput = 'raw_values')
            errs = errs + thing
        err = errs/5
        if scoring is None:
            return np.sqrt(err)
        else:
            return np.dot(np.sqrt(err), np.asarray(scoring))


    def eval_error(self, y, y_hat, scoring = None):
        if scoring is None:
            return np.sqrt(mean_squared_error(y, y_hat))
        else:
            return np.dot(np.sqrt(mean_squared_error(y, y_hat, multioutput = 'raw_values')), np.asarray(scoring))
    
    def save_model(self, name):
        self.model.save("./models/" + name + str(self.offset) + "-" + str(self.batch_size) + "-" + str(self.epochs))
        dump(self.X_scaler, open("./models/" + name + str(self.offset) + "-" + str(self.batch_size) + "-" + str(self.epochs) + "-X_scaler.pkl", 'wb'))
        dump(self.y_scaler, open("./models/" + name + str(self.offset) + "-" + str(self.batch_size) + "-" + str(self.epochs) + "-y_scaler.pkl", 'wb'))
    
    def load_model(self, name):
        self.model = keras.models.load_model(name)
        self.X_scaler = load(open(name + "-X_scaler.pkl", 'rb'))
        self.y_scaler = load(open(name + "-y_scaler.pkl", 'rb'))
