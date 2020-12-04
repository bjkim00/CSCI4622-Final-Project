import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataHandler():
    def __init__(self, beg = 1999, end = 2019,offset = 17,  split_by_pos = False, ignore_na = True, fill_mean = False, test_size = 0.2, min_num = 50):
        self.stats = ['PassingYds', 'PassingTD', 'Int', 'PassingAtt', 'Cmp', \
                      'RushingAtt', 'RushingYds', 'RushingTD', 'Rec', 'Tgt', 'ReceivingYds', 'ReceivingTD', 'FL']
        years = range(beg, end)
        data = pd.DataFrame()

        for y in years:
            for w in range(1, 18):
                new_data = pd.read_csv("data/weekly/" + str(y) + "/week" + str(w) + ".csv")
                new_data = new_data.assign(Year = y, Week = w)
                data = data.append(new_data)

        for i in range(1, offset + 1):
            data[[s + '-' + str(i) for s in self.stats]] = data.groupby('Player')[self.stats].shift(i)
        if ignore_na:
            data = data.dropna()
        else:
            if fill_mean:
                data = data.fillna(data.mean())
            else:
                data = data.fillna(0)
            
        data = data.reset_index()
        cols = data.columns.tolist()
        cols.remove("PPRFantasyPoints")
        cols.remove("StandardFantasyPoints") 
        cols.remove("HalfPPRFantasyPoints")
        cols.remove("Year")
        cols.remove("Week")
        
        cols = ["Year", "Week"] + cols[1:]
        data = data[cols]

        if split_by_pos:
            self.data = {}
            self.info_train, self.info_test, self.X_train, self.X_test, self.y_train, self.y_test = {}, {}, {}, {}, {}, {}
            for pos in data.Pos.unique():
                if len(data[data['Pos'] == pos]) >= min_num:
                    self.data[pos] = data[data['Pos'] == pos]
                    info, X, y = self.data[pos].iloc[:, :5], self.data[pos].iloc[:, 18:], self.data[pos].iloc[:, 5:18]
                    self.info_train[pos], self.info_test[pos], self.X_train[pos], self.X_test[pos], self.y_train[pos], self.y_test[pos] = train_test_split(info, X, y, test_size = test_size)   
        else:
            self.data = data
            info, X, y = self.data.iloc[:, :5], self.data.iloc[:, 18:], self.data.iloc[:, 5:18]
            self.info_train, self.info_test, self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(info, X, y, test_size = test_size)