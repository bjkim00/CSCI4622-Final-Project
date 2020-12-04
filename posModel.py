from sklearn.linear_model import Lasso
from fantasyModel import FantasyModel
from DataHandler import DataHandler

class posModel(DataHandler):
    def __init__(self, pos = None, year_range = [1999, 2019], offset = 17, alpha = 1.0):
        DataHandler.__init__(self, beg = year_range[0], end = year_range[1], offset = offset, split_by_pos = False,ignore_na = True, test_size = 0.2)
        self.alpha = alpha
        self.models = {s: Lasso(alpha = self.alpha) for s in self.stats}
    
    def train(self, data):
        for s in self.stats:
            self.models[s].fit(self.data.iloc[:,18:], self.data[s])
    
    def predict(self, X):
        return {s: self.models[s].predict(X)[0] for s in self.stats}