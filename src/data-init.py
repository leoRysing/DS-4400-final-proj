"""
Matthew Keefer, Leroy Schaigorodsky, Alan Sourek, Ceara Zhang
DS 4400 / Hockey Game Analysis
Final Project
Date Created: 3/29/23
Last Updated: 4/6/2023
"""

import pandas as pd
import nhlstats
from nhlstats import list_plays, list_shifts
import numpy as np

df = pd.DataFrame(nhlstats.list_games('2022-09-01', None))
df = df[df['season'] == '20222023']

gameId = 2022010001

plays = pd.DataFrame(list_plays(gameId))
shifts = pd.DataFrame(list_shifts(gameId))

print(df.head())
print(len(df))

print(plays.columns)
print(set(plays["event_type"]))

print(plays[plays["event_type"] == "GOAL"])

## reference for team name / abbreviation
team_name_to_abbrev = {
    'Anaheim Ducks': "ANA", 'Arizona Coyotes': "ARI", 'Boston Bruins': "BOS", 'Buffalo Sabres': "BUF",
    'Calgary Flames': "CGY", 'Carolina Hurricanes': "CAR", 'Chicago Blackhawks': "CHI", 'Colorado Avalanche': "COL",
    'Columbus Blue Jackets': "CBJ", 'Dallas Stars': "DAL", 'Detroit Red Wings': "DET", 'Edmonton Oilers': "EDM",
    'Eisbaren Berlin': "EIS", 'Florida Panthers': "FLA", 'Los Angeles Kings': "LAK", 'Minnesota Wild': "MIN",
    'Montréal Canadiens': "MTL", 'Nashville Predators': "NSH", 'New Jersey Devils': "NJD", 'New York Islanders': "NYI",
    'New York Rangers': "NYR", 'Ottawa Senators': "OTT", 'Philadelphia Flyers': "PHI", 'Pittsburgh Penguins': "PIT",
    'SC Bern': "SCB", 'San Jose Sharks': "SJS", 'Seattle Kraken': "SEA", 'St. Louis Blues': "STL",
    'Tampa Bay Lightning': "TBL",
    'Team Atlantic': "TAT", 'Team Central': "TAC", 'Team Pacific': 'TAP', 'Team Metropolitan': "TAM",
    'Toronto Maple Leafs': "TOR", 'Vancouver Canucks': "VAN", 'Vegas Golden Knights': "VGK",
    'Washington Capitals': "WSH",
    'Winnipeg Jets': "WPG"
}


def home_shots(row):
    """
    given plays of the game, returns the total number of shots
    taken by home team in the game
    :param row: row of a dataframe
    :return: number of shots taken by home team
    """

    plays = pd.DataFrame(list_plays(row["game_id"]))
    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    home_df = plays[plays["team_for"].notnull()]
    home_df = home_df[home_df["team_for"].str.match(home, case=False) & home_df["is_shot"] == True]
    return home_df.shape[0]


def away_shots(row):
    """
    given plays of a game, returns the total number of shots
    taken by the away team
    :param row: row of dataframe
    :return: number of shots taken by away team
    """
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    away_df = plays[plays["team_for"].notnull()]
    away_df = away_df[away_df["team_for"].str.match(away, case=False) & away_df["is_shot"] == True]
    return away_df.shape[0]


def home_corsi(row):
    """

    :param row:
    :return:
    """
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    home_df = plays[plays["team_for"].notnull()]
    home_df = home_df[home_df["team_for"].str.match(home, case=False) & home_df["is_corsi"] == True]
    return home_df.shape[0]


def away_corsi(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    away_df = plays[plays["team_for"].notnull()]
    away_df = away_df[away_df["team_for"].str.match(away, case=False) & away_df["is_corsi"] == True]
    return away_df.shape[0]


def home_fenwick(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    home_df = plays[plays["team_for"].notnull()]
    home_df = home_df[home_df["team_for"].str.match(home, case=False) & home_df["is_fenwick"] == True]
    return home_df.shape[0]


def away_fenwick(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    away_df = plays[plays["team_for"].notnull()]
    away_df = away_df[away_df["team_for"].str.match(away, case=False) & away_df["is_fenwick"] == True]
    return away_df.shape[0]


def home_mean_x(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    home_df = plays[plays["team_for"].notnull()]
    home_df = home_df[home_df["team_for"].str.match(home, case=False) & home_df["x"].notnull()]
    mean = np.mean(home_df["x"])
    return mean


def away_mean_x(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    away_df = plays[plays["team_for"].notnull()]
    away_df = away_df[away_df["team_for"].str.match(away, case=False) & away_df["x"].notnull()]
    mean = np.mean(away_df["x"])
    return mean


def get_nhl_dataframe():
    data = pd.DataFrame(nhlstats.list_games('2022-09-01', None))
    return data


def addRows(data):
    data["away_corsi"] = data.apply(away_corsi, axis=1)
    data["home_corsi"] = data.apply(home_corsi, axis=1)
    data["away_shots"] = data.apply(away_shots, axis=1)
    data["home_shots"] = data.apply(home_shots, axis=1)
    data["away_fenwick"] = data.apply(away_fenwick, axis=1)
    data["home_fenwick"] = data.apply(home_fenwick, axis=1)
