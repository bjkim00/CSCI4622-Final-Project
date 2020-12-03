import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from fantasyModel import FantasyModel
from qbModel import qbLinModel
from wrModel import wrModel

def main():
    years = range(1999, 2020)
    data = pd.DataFrame()
    for y in years:
        for w in range(1, 18):
            new_data = pd.read_csv("data/weekly/" + str(y) + "/week" + str(w) + ".csv")
            new_data = new_data.assign(Year = y, Week = w)
            data = data.append(new_data)

    stats = ['PassingYds', 'PassingTD', 'Int', 'PassingAtt', 'Cmp', 'RushingAtt', 'RushingYds', 'RushingTD', 'Rec', 'Tgt', 'ReceivingYds', 'ReceivingTD', 'FL']
    for i in range(1, 18):
        data[[s + '-' + str(i) for s in stats]] = data.groupby('Player')[stats].shift(i)
    data = data.dropna()
    data = data.reset_index()
    cols = data.columns.tolist()
    cols.remove("Year")
    cols.remove("Week")
    cols.remove("PPRFantasyPoints")
    cols.remove("StandardFantasyPoints") 
    cols.remove("HalfPPRFantasyPoints")
    cols = ["Year", "Week"] + cols[1:-3]
    data = data[cols]

    # print(data.columns)

    # print(len(data))

    lm = LinearModel(year_range = [1999, 2018])
    lm.train()
    print(lm.models)

    print(lm.predict(np.asarray(data.iloc[-1000, 18:]).reshape(1, -1)))
    print(data.iloc[-1000, :18])

main()