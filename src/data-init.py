import pandas as pd
import nhlstats
from nhlstats import list_plays, list_shifts

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
    'SC Bern': "SCB", 'San Jose Sharks': "SJS",'Seattle Kraken': "SEA", 'St. Louis Blues': "STL", 'Tampa Bay Lightning': "TBL",
    'Team Atlantic': "TAT", 'Team Central': "TAC",
    'Toronto Maple Leafs': "TOR", 'Vancouver Canucks': "VAN", 'Vegas Golden Knights': "VGK", 'Washington Capitals': "WSH",
    'Winnipeg Jets': "WPG"
}


def home_shots(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    home_df = plays[plays["team_for"].notnull()]
    home_df = home_df[home_df["team_for"].str.match(home, case=False) & home_df["is_shot"] == True]
    return home_df.shape[0]


def away_shots(row):
    plays = pd.DataFrame(list_plays(row["game_id"]))

    home, away = team_name_to_abbrev[row["home_team"]], team_name_to_abbrev[row["away_team"]]
    away_df = plays[plays["team_for"].notnull()]
    away_df = away_df[away_df["team_for"].str.match(away, case=False) & away_df["is_shot"] == True]
    return away_df.shape[0]



def get_nhl_dataframe():
    data = pd.DataFrame(nhlstats.list_games('2022-09-01', None))
    return data

def home_shots():

