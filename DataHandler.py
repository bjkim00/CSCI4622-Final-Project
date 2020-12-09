import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataHandler():
    def __init__(self, beg = 1999, end = 2019, offset = 17, split_by_pos = False, ignore_na = True, fill_mean = False, test_size = 0.2, min_num = 150, include_all = False):
        self.stats = ['PassingYds', 'PassingTD', 'Int', 'PassingAtt', 'Cmp', \
                      'RushingAtt', 'RushingYds', 'RushingTD', 'Rec', 'Tgt', 'ReceivingYds', 'ReceivingTD', 'FL']
        years = range(beg, end+1)
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

        specified_data = ['QB', 'WR', 'RB']
        
        if split_by_pos:
            self.all_data = data
            self.data = {}
            self.info_train, self.info_test, self.X_train, self.X_test, self.y_train, self.y_test = {}, {}, {}, {}, {}, {}
            # self.X_train_normal, self.X_test_normal, self.y_train_normal, self.y_test_normal = {}, {}, {}, {}
            for pos in specified_data:
                if pos == 'WR':
                    self.data[pos] = data[(data['Pos'] == pos) | (data['Pos'] == 'TE')]
                elif pos == 'RB':
                    self.data[pos] = data[(data['Pos'] == pos) | (data['Pos'] == 'HB') | (data['Pos'] == 'FB')]
                else:
                    self.data[pos] = data[data['Pos'] == pos]
                info, X, y = self.data[pos].iloc[:, :5], self.data[pos].iloc[:, 18:], self.data[pos].iloc[:, 5:18]
                if include_all:
                    self.info_test[pos], self.X_test[pos], self.y_test[pos] = info, X, y
                else:
                    self.info_train[pos], self.info_test[pos], self.X_train[pos], self.X_test[pos], self.y_train[pos], self.y_test[pos] = train_test_split(info, X, y, test_size = float(test_size))
                # self.X_train_normal[pos] = X_scaler.fit_transform(self.X_train[pos])
                # self.X_test_normal[pos] = X_scaler.transform(self.X_test[pos])
                # self.y_train_normal[pos] = y_scaler.fit_transform(self.y_train[pos])
                # self.y_test_normal[pos] = y_scaler.transform(self.y_test[pos])
    
        else:
            self.data = data
            info, X, y = self.data.iloc[:, :5], self.data.iloc[:, 18:], self.data.iloc[:, 5:18]
            self.info_train, self.info_test, self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(info, X, y, test_size = test_size)