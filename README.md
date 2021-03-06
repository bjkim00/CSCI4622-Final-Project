### CSCI 4622 Final Project

Group Members: Adam Hoerger, Neel Katuri, BJ Kim

## Introduction

Fantasy Football is something many people do every year. The system of adjusting your team based off of the projected points or just gut feeling is up to the person, but you can't help but notice the projected scores for the week next to your players. More often than not, your players won't score exactly on their proejcted scores, frequently scoring lower than their projected scores, but also on good days, performing way above their projected scores. Thus, we decided to see if we could create a model to help predict scores better.

## Dataset

The dataset we're using the weekly dataset from https://www.fantasyfootballdatapros.com/csv_files. Due to the limitations presented by the dataset, we only have access for weekly data from 1999 onward, but it shouldn't be a big issue as most players from back then or before then are retired.

## Dependencies
* Pandas
* Numpy
* Matplotlib
* Keras LSTM Layer
* Keras Dense Layer
* Keras Sequential Model
* Sklearn
* Pickle

## How To Run
Once the repo is cloned, you can choose what team to predict scores for by entering names in the `test_team` variable in the `team_prediction.py` file. Then, in the same location as the cloned repo, running `python team_prediction.py` will provide the predicted scores for those players.

