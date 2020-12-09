import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense

from DataHandler import DataHandler
from predModel import predModel
from linModel import linModel


SCORING = [0.04, 4, -2, 0, 0, 0, 0.1, 6, 1, 0, 0.1, 6, -2]
response = input("Would you like to provide values?")
find_vals = False
if response.lower() == "no":
    find_vals = True
if find_vals:
    ### run linear model for various alpha
    dh = DataHandler(beg=1999, end = 2018, split_by_pos=True, offset=16, ignore_na = False, fill_mean = True)
    alpha_vals = np.logspace(-2, 0.5, 10)
    errs = {}
    best_alphas = []
    for pos in dh.X_train.keys():
        errs[pos] = []
        print("Running: " + str(pos) + " (" + str(len(dh.X_train[pos])) + " entries)")
        for alpha in alpha_vals:
            lin = linModel(alpha = alpha, max_iter=5000)
            err = np.mean(lin.cv_error(dh.X_train[pos], dh.y_train[pos], np.abs(SCORING)))
            errs[pos].append(err)
        best_alphas.append(alpha_vals[np.argmin(errs[pos])])
        plt.figure()
        plt.plot(alpha_vals, errs[pos])
        plt.title(pos + " Error vs. Alpha (Lasso Model)")
        plt.xscale('log')
        plt.xlabel("Alpha (logarithmic scale)")
        plt.ylabel("Point-Weighted Error")
        plt.savefig("./figures/linear_alpha_" + pos)
    best_alpha = np.mean(best_alphas)
    print("Best alpha:", best_alpha)

    ### run linear model for various offsets
    alpha = max(0.1, best_alpha)
    errs = {}
    offset_vals = np.arange(8, 25, 2)
    best_offsets = []
    dh = DataHandler(beg=1999, end = 2018, split_by_pos=True, offset=24, ignore_na = False, fill_mean = True)
    for pos in dh.X_train.keys():
        errs[pos] = []
        print("Running: " + str(pos) + " (" + str(len(dh.X_train[pos])) + " entries)")
        for offset in offset_vals:
            lin = linModel(alpha = alpha)
            err = np.mean(lin.cv_error(dh.X_train[pos].iloc[:, :offset*13], dh.y_train[pos], np.abs(SCORING)))
            errs[pos].append(err)
        best_offsets.append(offset_vals[np.argmin(errs[pos])])
        plt.figure()
        plt.plot(offset_vals, errs[pos])
        plt.title(pos + " Error vs. Offset (Lasso Model)")
        plt.xlabel("Offset (weeks)")
        plt.ylabel("Point-Weighted Error")
        plt.savefig("./figures/linear_offsets_" + pos)
    best_offset_linear = int(np.round(np.mean(best_offsets)))
    print("Best offset: ", best_offset_linear)
else:
    best_alpha = float(input("Alpha: "))
    best_offset_linear = int(input("Offset (Linear): "))
    best_offset_lstm = int(input("Offset (LSTM): "))
    dh = DataHandler(beg=1999, end = 2018, split_by_pos=True, offset=max(best_offset_linear, best_offset_lstm), ignore_na = False, fill_mean = True)

### evaluate linear test model for best alpha, offset
eval_errs_linear = []
for pos in dh.X_test.keys():
    print("Running: " + str(pos) + " (" + str(len(dh.X_train[pos])) + " entries)")
    lin = linModel(alpha = best_alpha)
    lin.train(dh.X_train[pos].iloc[:, :best_offset_linear*13], dh.y_train[pos])
    y_pred = lin.predict(dh.X_test[pos].iloc[:, :best_offset_linear*13])
    err = lin.eval_error(y_pred, dh.y_test[pos], np.abs(SCORING))
    eval_errs_linear.append(err)

if find_vals:
    ### run lstm model for various offsets
    offset_vals = np.arange(8, 25, 4)
    best_offsets = []
    errs = {}
    for pos in dh.X_train.keys():
        errs[pos] = []
        print("Running: " + str(pos) + " (" + str(len(dh.X_train[pos])) + " entries)")
        for offset in offset_vals:
            pred = predModel(offset = offset, epochs=75)
            err = pred.cv_error(dh.X_train[pos].iloc[:, :offset*13], dh.y_train[pos], np.abs(SCORING))
            errs[pos].append(err)
        best_offsets.append(offset_vals[np.argmin(errs[pos])])
        plt.figure()
        plt.plot(offset_vals, errs[pos])
        plt.title(pos + " Error vs. Offset (LSTM Model)")
        plt.xlabel("Offset (weeks)")
        plt.ylabel("Point-Weighted Error")
        plt.savefig("./figures/lstm_offsets_" + pos)
    best_offset_lstm = int(np.round(np.mean(best_offsets)))
    print("Best LSTM Offset: ", best_offset_lstm)

### evaluate lstm model for best offset
eval_errs_lstm = []
for pos in dh.X_train.keys():
    print("Running: " + str(pos) + " (" + str(len(dh.X_train[pos])) + " entries)")
    pred = predModel(offset = best_offset_lstm)
    pred.train(dh.X_train[pos].iloc[:, :best_offset_lstm*13], dh.y_train[pos])
    y_pred = pred.predict(dh.X_test[pos].iloc[:, :best_offset_lstm*13])
    err = pred.eval_error(y_pred, dh.y_test[pos], np.abs(SCORING))
    eval_errs_lstm.append(err)
    pred.save_model(pos)

index = [pos for pos in dh.X_train.keys()]
df = pd.DataFrame({'linear': eval_errs_linear, 'lstm': eval_errs_lstm}, index = index)
df.plot.bar()
plt.title("Errors on Test Data")
plt.ylabel("Point-Weighted Error")
plt.savefig("./figures/test_eval")