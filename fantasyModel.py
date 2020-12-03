import pandas as pd
import numpy as np

class FantasyModel():
    def __init__(self, pos = None, year_range = [1999, 2019], offset = 17):
        self.position = pos
        self.beginning = year_range[0]
        self.end = year_range[1]
        self.offset = offset
        self.stats = ['PassingYds', 'PassingTD', 'Int', 'PassingAtt', 'Cmp', \
                      'RushingAtt', 'RushingYds', 'RushingTD', 'Rec', 'Tgt', 'ReceivingYds', 'ReceivingTD', 'FL']
        self.load_data()
        
    def load_data(self):
        years = range(self.beginning, self.end)
        data = pd.DataFrame()
        for y in years:
            for w in range(1, 18):
                new_data = pd.read_csv("data/weekly/" + str(y) + "/week" + str(w) + ".csv")
                new_data = new_data.assign(Year = y, Week = w)
                data = data.append(new_data)
        if not self.position is None:
            data = data[self.position in data['Pos']]
        for i in range(1, self.offset + 1):
            data[[s + '-' + str(i) for s in self.stats]] = data.groupby('Player')[self.stats].shift(i)
        data = data.dropna()
        data = data.reset_index()
        cols = data.columns.tolist()
        cols.remove("PPRFantasyPoints")
        cols.remove("StandardFantasyPoints") 
        cols.remove("HalfPPRFantasyPoints")
        cols.remove("Year")
        cols.remove("Week")
        cols = ["Year", "Week"] + cols[1:-3]
        self.data = data[cols]