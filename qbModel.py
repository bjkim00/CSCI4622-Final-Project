from sklearn.linear_model import Lasso
from fantasyModel import FantasyModel

class qbLinModel(FantasyModel):
    def __init__(self, pos = None, year_range = [1999, 2019], offset = 17, alpha = 1.0):
        FantasyModel.__init__(self, pos = pos, year_range = year_range, offset = offset)
        self.alpha = alpha
        self.models = {s: Lasso(alpha = self.alpha) for s in self.stats}
    
    def train(self):
        for s in self.stats:
            self.models[s].fit(self.data.iloc[:,18:], self.data[s])
    
    def predict(self, X):
        return {s: self.models[s].predict(X)[0] for s in self.stats}