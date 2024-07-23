import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('NBA_Historical_Odds/realtime_pipeline')
from boxes_betting_builder import *

#gets player box scores for a given player and year

def get_player_boxes(player, year):
    player = player.lower()
    player_code = player[player.index(' ')+1:player.index(' ')+6] + player[:2] + '01'
    player_url = f'https://www.basketball-reference.com/players/g/{player_code}/gamelog/{year}'
    response = requests.get(player_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    tables = soup.find_all('table')
    if len(tables)==0:
        player_code = player[player.index(' ')+1:player.index(' ')+6] + player[:2] + '02'
    player_url = f'https://www.basketball-reference.com/players/g/{player_code}/gamelog/{year}'
    response = requests.get(player_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    tables = soup.find_all('table')
    table = tables[7]
    # Check if the table was found
    if table:
        # Parse the table with pandas to get a DataFrame
        df = pd.read_html(str(table))[0]
        df = df[df['Date'] != 'Date']
        df['PTS'].replace('Inactive', '-10', inplace=True)
        df['AST'].replace('Inactive', '-10', inplace=True)
        df['TRB'].replace('Inactive', '-10', inplace=True)
        df['MP'] = df['MP'].replace('Inactive', '-10', inplace=True)
        df = df[df['MP'] != 'Inactive']
        df = df[df['PTS'] != 'Did Not Dress']
        df = df[df['PTS'] != 'Did Not Play']
        df[['PTS','TRB','AST']] = df[['PTS','TRB','AST']].astype(int)
        
        df['PTS+AST'] = df['PTS'] + df['AST']
        df['REB+AST'] = df['AST'] + df['TRB']
        df['PTS+REB+AST'] = df['PTS'] + df['AST'] + df['TRB']

        

        return df
    else:
        print('Table not found')
        return None


#takes in player boxes and season lines and returns merged frame

def map_games(player_boxes, season_lines,relevant_stat):

    season_lines = season_lines[(season_lines['away_abbrev'] == player_team) | (season_lines['home_abbrev']==player_team)]
    season_lines = season_lines.reset_index()


    season_lines['away_ml'] = pd.to_numeric(season_lines['away_ml'], errors='coerce')
    season_lines['home_ml'] = pd.to_numeric(season_lines['home_ml'], errors='coerce')

    season_lines['home_favor'] = (season_lines['home_ml'] < season_lines['away_ml']).astype(int)
    season_lines['away_favor'] = (season_lines['away_ml'] < season_lines['home_ml']).astype(int)

    season_lines.loc[((season_lines['away_abbrev'] == player_team) & (season_lines['away_favor'] == 1)) |
                    ((season_lines['home_abbrev'] == player_team) & (season_lines['away_favor'] == 1)) ,'team_favor'] = 1
    season_lines['team_favor']  = season_lines['team_favor'].fillna(0)

    season_lines['spread'] = abs(season_lines['spread'])

    season_lines.loc[season_lines['team_favor'] == 1, 'team_spread'] = season_lines['spread']
    season_lines.loc[season_lines['team_favor'] != 1, 'team_spread'] = -1* season_lines['spread']



    minutes = 'MP'
   

    rel_stat_boxes = player_boxes[['Date','Opp',minutes,relevant_stat]]
    rel_stat_boxes = rel_stat_boxes.reset_index()

    all_data = pd.concat([season_lines, rel_stat_boxes],axis=1)

    all_data = all_data[all_data[relevant_stat] >= 0]

    return all_data

#takes in all_data from map games and then gets
def get_similar_games(all_data, relevant_stat, weights,n_similar,new_event):


    game_predicts = all_data[['team_spread','o/u',relevant_stat]]

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(game_predicts[['team_spread', 'o/u']]), columns=game_predicts.columns[:2])

    df_scaled['weighted_total'] = df_scaled['o/u'] * weights['o/u']
    df_scaled['weighted_spread'] = df_scaled['team_spread'] * weights['team_spread']

    new_event_scaled = scaler.transform([[new_event['team_spread'], new_event['o/u']]])

    new_event_weighted_total = new_event_scaled[0][0] * weights['team_spread']
    new_event_weighted_diff = new_event_scaled[0][1] * weights['o/u']

    df_scaled['similarity_score'] = ((df_scaled['weighted_total'] - new_event_weighted_total) ** 2 + (df_scaled['weighted_spread'] - new_event_weighted_diff) ** 2) ** 0.5

    all_data['similarity_score'] = df_scaled['similarity_score']
    all_data_sorted = all_data.sort_values(by='similarity_score')

    
    sim_vals = all_data_sorted.head(n_similar)[relevant_stat].values

    return sim_vals

#creates visual distirbution of bootstrapped similar games
#returns summary stast of distribution

def bootstrap_analysis(data, n_bootstraps=250):
    
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_stds = []
    
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
        bootstrap_medians.append(np.median(bootstrap_sample))
        bootstrap_stds.append(np.std(bootstrap_sample))
    
    # Visualization
    plt.hist(bootstrap_means, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Bootstrap Means')
    plt.xlabel('Mean Points')
    plt.ylabel('Frequency')
    plt.show()
    
    # Calculating overall summary statistics
    summary_stats = {
        'mean_of_means': np.mean(bootstrap_means),
        'median_of_medians': np.median(bootstrap_medians),
        'std_of_means': np.std(bootstrap_means),
        'overall_mean': np.mean(data),
        'overall_median': np.median(data),
        'overall_std': np.std(data)
    }
    
    return summary_stats






"""prev_date = '2023-10-24'   #last date that there is a sheet from
start_date = datetime(2023, 10, 24)
end_date = datetime(2024, 3, 13)

dates = []

current_date = start_date
while current_date <= end_date:
    dates.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

end_date = end_date.strftime('%Y-%m-%d')
dates.remove('2024-02-15')
dates.remove('2024-02-16')
dates.remove('2024-02-18')



#for pulling all 2024 lines
new_betting_lines = get_betting_lines(dates = dates, prev_date = '2023-24', just_today= True)


"""


#todo just generalize the season lines thing for a given player and year
player_team = 'PHX'
player_boxes = get_player_boxes(player='Kevin Durant',year=2024)


season_lines = pd.read_csv('all_odds/20240314_odds.csv')

betting_teams = sorted(season_lines.homeTeam.unique())
player_teams = sorted(np.append(player_boxes.Opp.unique(),player_team))
#player_teams.remove('Opp')

betting_player_dict = {'Atlanta': 'ATL',
 'Boston': 'BOS',
 'Brooklyn': 'BRK',
 'Charlotte': 'CHO',
 'Chicago': 'CHI',
 'Cleveland': 'CLE',
 'Dallas': 'DAL',
 'Denver': 'DEN',
 'Detroit': 'DET',
 'Golden State': 'GSW',
 'Houston': 'HOU',
 'Indiana': 'IND',
 'L.A. Clippers': 'LAC',
 'L.A. Lakers': 'LAL',
 'Memphis': 'MEM',
 'Miami': 'MIA',
 'Milwaukee': 'MIL',
 'Minnesota': 'MIN',
 'New Orleans': 'NOP',
 'New York': 'NYK',
 'Oklahoma City': 'OKC',
 'Orlando': 'ORL',
 'Philadelphia': 'PHI',
 'Phoenix': 'PHO',
 'Portland': 'POR',
 'Sacramento': 'SAC',
 'San Antonio': 'SAS',
 'Toronto': 'TOR',
 'Utah': 'UTA',
 'Washington': 'WAS'}

season_lines['away_abbrev'] = season_lines['awayTeam'].map(betting_player_dict)
season_lines['home_abbrev'] = season_lines['homeTeam'].map(betting_player_dict)

relevant_stats = ['PTS','AST','TRB','PTS+AST','REB+AST','PTS+REB+AST']

all_data = map_games(player_boxes=player_boxes,season_lines=season_lines, relevant_stat='PTS')
relevant_stat='PTS'
"""
all_data['o/u'].describe()

all_data[all_data['o/u'] <= 229][relevant_stat].hist(bins=50)
all_data[all_data['o/u'] >= 235][relevant_stat].hist(bins=50)"""

#NOW GET CLOSEST GAMES TO GIVEN TOTAL AND SPREAD BY DIFF WEIGHTS
#IF TEAM FAVORED, SPREAD IS POSITIVE, UNDERDOG IS NEGATIVE

all_data['team_spread'].hist(bins=50)
spread_weight = 0.7

weights= {'o/u':1-spread_weight ,'team_spread':spread_weight}
new_event = {'o/u': 226, 'team_spread': 5.50}

sim_points = get_similar_games(all_data,relevant_stat,weights,n_similar=15, new_event=new_event)

similar_summary = bootstrap_analysis(sim_points, n_bootstraps=2500)
sim_sum_df = pd.DataFrame(similar_summary.values(),index=similar_summary.keys())


all_games_stats = all_data[relevant_stat].values
all_games_summary = bootstrap_analysis(all_games_stats, n_bootstraps=2500)
all_sum_df = pd.DataFrame(all_games_summary.values(),index=all_games_summary.keys())

sim_sum_df - all_sum_df

all_data[all_data['o/u'] <= 229][relevant_stat].median()
all_data[all_data['o/u'] <= 229][relevant_stat].hist(bins=50)

all_data[all_data['o/u'] >= 235][relevant_stat].median()
all_data[all_data['o/u'] >= 235][relevant_stat].hist(bins=50)

all_data[all_data['team_spread'] <= 6][relevant_stat].describe()
all_data[all_data['team_spread'] >6][relevant_stat].describe()

