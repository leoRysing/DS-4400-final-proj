import pandas as pd
import nhlstats

df = pd.DataFrame(nhlstats.list_games('2022-09-01', None))
df = df[df['season'] == '20222023']
print(df.head())
print(len(df))