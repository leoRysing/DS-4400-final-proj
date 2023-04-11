"""
Matthew Keefer, Leroy Schaigorodsky, Alan Sourek, Ceara Zhang
DS 4400 / Hockey Game Analysis
Final Project
Date Created: 3/29/23
Last Updated: 3/29/23
"""

from decisiontree import dtree, entropy, bag
import pandas as pd
import numpy as np
from algorithms import predict, metrics, collapseToBin, predict_bag


# these two lists are used for column selection
listCols = ['date', 'home_team', 'away_team', 'home_shots', 'away_shots', 'away_corsi',
       'home_corsi', 'home_fenwick', 'away_fenwick', 'home_mean_x',
       'away_mean_x', 'home_mean_y', 'away_mean_y', 'home_penalties',
       'away_penalties', 'home_hits', 'away_hits', 'home_takeaways',
       'away_takeaways']

selected_cols = [
       'home_team', 'away_team', 'home_shots_prop', 'away_shots_prop',
       'home_corsi_prop', 'away_corsi_prop', 'home_fenwick_prop',
       'away_fenwick_prop', 'home_penalties_prop', 'away_penalties_prop',
       'home_hits_prop', 'away_hits_prop', 'home_takeaways_prop',
       'away_takeaways_prop', 'game_end'
]


# get the score result - translate the home / away scores to see which team won
def get_score_result(x):
    home, away = x["home_score"], x["away_score"]
    if home > away:
        return "home win"
    elif away > home:
        return "away win"
    else:
        return "tie"

# method used for setup, gets called in main to be run
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    df = pd.read_csv("../data/team_data_v4.csv")


    # load the most recent csv, and drop the extra column that gets added (something with index being read as a column
    df = pd.read_csv("../data/team_data_v4.csv")
    df.drop(["Unnamed: 0"], inplace=True, axis=1)
    columns = list(df.columns)
    print(df)

    # make a dtree model, and make predictions
    model = dtree(df, entropy, columns[-1], 8, 4, 0.0)
    predictions = predict(model, df)
    actuals = df.game_end

    # make a function object that can vectorize the string classes down to
    # 0 and 1 values
    apply = np.vectorize(lambda x: collapseToBin("home win", x))

    predicts = apply(predictions)
    acts = apply(actuals)

    # get the metrics of this model
    measures = metrics(acts, predicts)
    print(measures)

    # try bagging this df
    bag_model = bag(df, entropy, columns[-1], 8, 4, 0.0)
    predictions = predict_bag(bag_model, df)
    predicts = apply(predictions)

    measures = metrics(acts, predicts)
    print(measures)



    # setup the new portion of the dataframe, with props
    """
    df2["home_shots_prop"] = df.apply(lambda x: x["home_shots"] / (x["home_shots"] + x["away_shots"] + 1), axis=1)
    df2["away_shots_prop"] = df.apply(lambda x: x["away_shots"] / (x["home_shots"] + x["away_shots"] + 1), axis=1)
    df2["home_corsi_prop"] = df.apply(lambda x: x["home_corsi"] / (x["home_corsi"] + x["away_corsi"] + 1), axis=1)
    df2["away_corsi_prop"] = df.apply(lambda x: x["away_corsi"] / (x["home_corsi"] + x["away_corsi"] + 1), axis=1)
    df2["home_fenwick_prop"] = df.apply(lambda x: x["home_fenwick"] / (x["home_fenwick"] + x["away_fenwick"] + 1), axis=1)
    df2["away_fenwick_prop"] = df.apply(lambda x: x["away_fenwick"] / (x["home_fenwick"] + x["away_fenwick"] + 1), axis=1)
    df2["home_penalties_prop"] = df.apply(lambda x: x["home_penalties"] / (x["home_penalties"] + x["away_penalties"] + 1), axis=1)
    df2["away_penalties_prop"] = df.apply(lambda x: x["away_penalties"] / (x["home_penalties"] + x["away_penalties"] + 1), axis=1)
    df2["home_hits_prop"] = df.apply(lambda x: x["home_hits"] / (x["home_hits"] + x["away_hits"] + 1), axis=1)
    df2["away_hits_prop"] = df.apply(lambda x: x["away_hits"] / (x["home_hits"] + x["away_hits"] + 1), axis=1)
    df2["home_takeaways_prop"] = df.apply(lambda x: x["home_takeaways"] / (x["home_takeaways"] + x["away_takeaways"] + 1), axis=1)
    df2["away_takeaways_prop"] = df.apply(lambda x: x["away_takeaways"] / (x["home_takeaways"] + x["away_takeaways"] + 1), axis=1)
    
    # adding the game_end field;
    df2["game_end"] = df.apply(get_score_result, axis=1)

    df3 = df2[selected_cols]
    """

    #df3.to_csv("../data/team_data_v4.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
