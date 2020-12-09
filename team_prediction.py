import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense

from DataHandler import DataHandler
from predModel import predModel
from linModel import linModel

scoring = [0.04, 4, -2, 0, 0, 0, 0.1, 6, 1, 0, 0.1, 6, -2] # These are the standard PPR scoring weights
stats = ['PassingYds', 'PassingTD', 'Int', 'PassingAtt', 'Cmp', \
                      'RushingAtt', 'RushingYds', 'RushingTD', 'Rec', 'Tgt', 'ReceivingYds', 'ReceivingTD', 'FL'] # These are the stats we care about
dh = DataHandler(beg=1999, end = 2018, split_by_pos=True, offset=24, ignore_na = False, fill_mean = True, include_all=True)

best_alpha = 0.12 # This value was determined from model_optimization.py
best_offset_linear = best_offset_lstm = 19 # This value was determined from model_optimization.py
lin = {}
for pos in dh.X_test.keys():
    lin[pos] = linModel(alpha = best_alpha)
    lin[pos].train(dh.X_test[pos].iloc[:, :best_offset_linear*13], dh.y_test[pos])
                      
dh = DataHandler(beg=1999, end = 2019, split_by_pos=True, offset=24, ignore_na = False, fill_mean = True, include_all=True)
test_team = {'QB': ['Patrick Mahomes'], 
             'RB': ['Alvin Kamara', 'Christian McCaffrey', 'Saquon Barkley'], 
             'WR': ['Michael Thomas', 'Travis Kelce', 'Julio Jones']}


##### Predict what players on test team would score for each week #####
##### Output predicted points, actual points, and totals for both #####

test_team_week = {}
total_weeks = np.zeros(17)

pred = predModel(offset = 16, epochs = 1)
for pos in test_team.keys():
    ind = dh.info_test[pos]['Year'] == 2019
    temp_info = dh.info_test[pos][ind].iloc[:, :best_offset_lstm*13]
    temp_X = dh.X_test[pos][ind].iloc[:, :best_offset_lstm*13]
    temp_y = dh.y_test[pos][ind].iloc[:, :best_offset_lstm*13]
    pred.load_model("./models/" + pos + str(best_offset_lstm) + "-32-100")
    for player in test_team[pos]:
        info_player, y_pred_stats, info_indices = lin[pos].predict_player(temp_info, temp_X, player, ret_info=True)
        y_linear_score = np.dot(y_pred_stats, np.asarray(scoring).T)
        
        info_player, y_pred_stats, info_indices = pred.predict_player(temp_info, temp_X, player, ret_info=True)
        y_lstm_score = np.dot(y_pred_stats, np.asarray(scoring).T)
        
        actual_score = np.dot(np.asarray(temp_y.loc[info_indices]), np.asarray(scoring).T)
        temp = pd.concat([info_player, pd.DataFrame({'linear_prediction':y_linear_score}).set_index(info_player.index), pd.DataFrame({'lstm_prediction':y_lstm_score}).set_index(info_player.index), pd.DataFrame({'actual':actual_score}).set_index(info_player.index)], axis = 1)

        for w in range(1, 18):
            # This case handles if the player did not play that week
            if not w in temp_info[temp_info['Player'] == player].Week.unique():
                player_info = (player, pos, 0, 0, 0)
            else:
                player_info = (player, pos, round(temp[temp['Week'] == w].iloc[0]['linear_prediction'], 1), round(temp[temp['Week'] == w].iloc[0]['lstm_prediction'], 1), round(temp[temp['Week'] == w].iloc[0]['actual'], 1))

            if w in test_team_week:
                test_team_week[w].append(player_info)
            else:
                test_team_week[w] = [player_info]

# The ouput reads as follows:
# Position: Player name, Predicted Points: Linear prediction/LSTM prediction, Actual Point: True Values

for w in range(1, 18):
    print("Week ",w,":")
    total_pred_linear = 0
    total_pred_lstm = 0
    total_real = 0
    for player in test_team_week[w]:
        print("{0}: {1}, Predicted Points: {2}/{3}, Actual Points: {4}".format(player[1], player[0], player[2], player[3], player[4]))
        total_pred_linear += float(player[2])
        total_pred_lstm += float(player[3])
        total_real += float(player[4])
    print("Total Predicted Points: {}/{}".format(round(total_pred_linear, 1), round(total_pred_lstm, 1)))
    print("Total Actual Points: ", round(total_real, 1))
    print("\n")


##### This will print the best possible teams for each week #####
##### Note: this is based off of the actual scores achieved #####

print("---------------------------------")
print("BEST POSSIBLE TEAMS FOR EACH WEEK")
print("---------------------------------")
best_team = {}

for pos in dh.data.keys():  
    # Set up new dataframe for only the year 2019 with specific position                    
    year_df = dh.data[pos]
    year_df = year_df.loc[year_df['Year'] == 2019]
    ppr_score = np.zeros(year_df.shape[0])
    year_df['PPR'] = ppr_score
    year_df = year_df 

    # Adding the PPR scores of each player
    for i, stat in enumerate(stats):
        year_df['PPR'] += year_df[stat] * scoring[i]

    # Go through each week for the specified position to find best performing player
    # Note: We only get the top 1 quarterback as you're only allowed one per team
    # cont.: and 3 running backs/wide receivers as they could go in the flex slot
    # Note 2: Tight ends were considered under the same category as wide receivers
    for w in range(1, 18):
        week_df = year_df.loc[year_df['Week'] == w]
        for i in range(1, 4):
            index = week_df[week_df.PPR == week_df.PPR.max()].index

            highest_ppr = week_df.PPR.max()
            best_player = week_df[week_df.PPR == highest_ppr].to_numpy()

            best_player_name = best_player[0][2]
            position = best_player[0][3]
            points = round(best_player[0][-1], 1)

            week_df.drop(index, inplace=True)

            best_player_info = (best_player_name, position, points)

            if w in best_team:
                best_team[w].append(best_player_info)
            else:
                best_team[w] = [best_player_info]

            if pos == 'QB':
                break

for best in best_team:
    print("Week ",best,":")
    total = 0
    for player in best_team[best]:
        print("{0}: {1}, Points: {2}".format(player[1], player[0], player[2]))
        total += player[2]
    print("Total Points: ", round(total, 1))
    print("\n")
