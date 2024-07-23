import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error



##############################################
  # DATA PREPARATION
##############################################


old_data = pd.read_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/gpt_builds/all_data_with_smaller_ma_season_stats.csv')

data = pd.read_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/all_odds/all_data_openers_poss_20240216.csv')


data['datetime_date'] = pd.to_datetime(data['finalDate'])
data['month'] = data['datetime_date'].dt.month
data = data[data['regSeason'] == 1]
data['favorite_dollars_risked'] = data['favorite_moneyline'].abs()
data['underdog_dollars_risked'] = 100


data['total_points'] = data['finalAway'] + data['finalHome']
data['total_difference'] = data['total_points'] - data['o/u']

def filter_data(data):

    data = data[(data['12_game_home_3pt_rate'] != 10) &
                                        (data['12_game_away_3pt_rate'] != 10) &
                                        (data['rolling_home_defense_relative_3pt_rate'] < 1.5) &
                                        (data['rolling_away_defense_relative_3pt_rate'] <1.5) &
                                        (data['away_total_shot_attempts'] > 60) &
                                        (data['home_total_shot_attempts'] > 60) & 
                                        (data['spread_difference'] != 0) &
                                        (data['spread'] <= 20) &
                                        (data['o/u'] <= 275) &
                                        (data['o/u'] >= 150) &
                                        (data['home_points_per_possession'] <= 1.75) &
                                        (data['away_points_per_possession'] <= 1.75)]


    data = data[data['season'] >= 2016]
    return data



###############################
# playoff work
##############################

playoffs = data[data['regSeason'] == 0]





###############################
# high spreads by month
##############################

filtered_data = filter_data(data)

high_spreads = filtered_data[filtered_data['spread']>= 13]
high_spreads.groupby('month')['favorite_spread_cover'].mean()


###############################
# what stat do away teams do worse
##############################

filtered_data = filter_data(data)

stats = ['points_per_possession', '2pt_percent','3pt_percent','off_reb_percentage','turnover_percentage']

stat_results = {}
for stat in stats:
  for location in ['home','away']:
    filtered_data['game_vs_avg'] = filtered_data[f'{location}_{stat}'] - filtered_data[f'12_game_{location}_{stat}']
    stat_results[f'{location}_{stat}'] = filtered_data['game_vs_avg'].median()










###############################
# given loss spread coverage ratios
##############################


data['s_bucket'] = pd.qcut(data['spread'], q=15, labels=False)

data.groupby('s_bucket')['favorite_spread_cover'].mean()


underdogs_lose =  data[((data['home_win'] == 1) & (data['home_favor'] != 1)) |
                      ((data['away_win'] == 1) & (data['away_favor'] != 1))]
favorites_win = data[((data['home_win'] == 1) & (data['home_favor'] == 1)) |
                      ((data['away_win'] == 1) & (data['away_favor'] == 1))]

favorites_win.groupby('s_bucket')['favorite_spread_cover'].mean()
favorites_win.groupby('s_bucket')['underdog_spread_cover'].mean()
favorites_win.groupby('s_bucket')['underdog_spread_cover'].count()



favorites_win[favorites_win['s_bucket'] == 4]['spread'].value_counts()



###############################
# b2b by spread bucket
##############################
data = data[data['away_rest'] <= 8]
data = data[data['home_rest'] <= 8]

data['s_bucket'] = pd.qcut(data['spread'], q=5, labels=False)

data.groupby('s_bucket')['spread'].median()


data.away_rest.hist(bins=10)

away_b2b = data[data['away_rest'] == 0]

away_b2b.groupby('s_bucket')['away_spread_cover'].mean()
away_b2b.groupby('s_bucket')['away_spread_cover'].count()


home_b2b = data[data['home_rest'] == 0]

home_b2b.groupby('s_bucket')['home_spread_cover'].mean()
home_b2b.groupby('s_bucket')['home_spread_cover'].count()



new_data = data[data['season'] >= 2019]
new_data['total_bucket'] = pd.qcut(new_data['o/u'], q=5, labels=False)

new_data.groupby('total_bucket')['o/u'].median()


new_data.away_rest.hist(bins=10)

away_b2b = new_data[new_data['away_rest'] == 0]

away_b2b.groupby('total_bucket')['over_under_result'].mean()
away_b2b.groupby('total_bucket')['over_under_result'].count()


home_b2b = new_data[new_data['home_rest'] == 0]

home_b2b.groupby('total_bucket')['over_under_result'].mean()
home_b2b.groupby('total_bucket')['over_under_result'].count()








###############################
# seeing spread change open to clsoe
##############################

data = data[data['regSeason'] == 1]
data = data[data['season'] >= 2018]

data['spread_change'] = abs(data['spread'] - data['opening_spread'])

data = data[data['spread_change'] <= 15]

###############################
# spread difference by month and year
##############################


data = data[data['regSeason'] == 1]
data = data[data['season'] >= 2018]
data = data[data['month'].isin([10,11,12,1,2,3,4,5])]
data.groupby(['season','month'])['spread_difference'].quantile(0.5)#.loc[2016].values[0]
data.groupby(['month'])['total_difference'].quantile(0.25)#.loc[2016].values[0]



###############################
# power rank calc + log 5
##############################
filtered_data = filter_data(data)

filtered_data['home_net_rtg'] = filtered_data['12_game_home_points_per_possession'] - filtered_data['12_game_home_opponent_points_per_possession']
filtered_data['away_net_rtg'] = filtered_data['12_game_away_points_per_possession'] - filtered_data['12_game_away_opponent_points_per_possession']

filtered_data['12_game_home_offense_points'] = filtered_data['12_game_home_points_per_possession'] * filtered_data['12_game_home_possessions'] * 12
filtered_data['12_game_home_defense_points'] = filtered_data['12_game_home_opponent_points_per_possession'] * filtered_data['12_game_home_possessions'] * 12

filtered_data['home_pyth_exp'] = (filtered_data['12_game_home_offense_points'] ** 13.91) / ((filtered_data['12_game_home_offense_points'] ** 13.91) + (filtered_data['12_game_home_defense_points']**13.91 ))


filtered_data['12_game_away_offense_points'] = filtered_data['12_game_away_points_per_possession'] * filtered_data['12_game_away_possessions'] * 12
filtered_data['12_game_away_defense_points'] = filtered_data['12_game_away_opponent_points_per_possession'] * filtered_data['12_game_away_possessions'] * 12

filtered_data['away_pyth_exp'] = (filtered_data['12_game_away_offense_points'] ** 13.91) / ((filtered_data['12_game_away_offense_points'] ** 13.91) + (filtered_data['12_game_home_defense_points']**13.91 ))

filtered_data['home_win_away_lose'] = filtered_data['home_pyth_exp'] * (1-filtered_data['away_pyth_exp'])
filtered_data['away_win_home_lose'] = filtered_data['away_pyth_exp'] * (1-filtered_data['home_pyth_exp'])

filtered_data['home_log5'] = filtered_data['home_win_away_lose'] / ((filtered_data['home_win_away_lose']) + filtered_data['away_win_home_lose'] )
filtered_data['home_log5'] = filtered_data['home_log5'] + 0.02

filtered_data['away_log5'] = filtered_data['away_win_home_lose'] / ((filtered_data['home_win_away_lose']) + filtered_data['away_win_home_lose'] )
filtered_data['away_log5'] = filtered_data['away_log5'] - 0.02


diff = 0.4

filtered_data.loc[(filtered_data['home_log5'] - filtered_data['home_win_prob']) >= diff, 'bet_home'] = 1
filtered_data['bet_home'] = filtered_data['bet_home'].fillna(0)

filtered_data[filtered_data['bet_home'] == 1].home_spread_cover.mean()
filtered_data[filtered_data['bet_home'] == 1][['awayTeam','homeTeam','spread','finalAway','finalHome','home_log5','away_log5','home_win_prob']]


filtered_data.loc[(filtered_data['away_log5'] - filtered_data['away_win_prob']) >= diff, 'bet_away'] = 1
filtered_data['bet_away'] = filtered_data['bet_away'].fillna(0)

filtered_data[filtered_data['bet_away'] == 1].away_spread_cover.mean()
filtered_data[filtered_data['bet_away'] == 1].shape



###############################
# mini regressions for points
##############################

data = filter_data(data)

data['const'] = 1
offense_col = '12_game_home_points_per_possession'
defense_col = '12_game_away_opponent_points_per_possession'
pace_col = '12_game_expected_pace'
data['home_offense_exp_points'] = data[offense_col] * data[pace_col]
data['away_defense_exp_points'] = data[defense_col] * data[pace_col]
data['home_exp_points'] = 0.5 * data['home_offense_exp_points'] + 0.5 * data['away_defense_exp_points']
betting_col = 'home_team_total'

X = data[['home_team_total','const']]
y = data['finalHome']


# Creating a linear regression model
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the model using the training sets
model.fit(X_train, y_train)

# Making predictions using the testing set
y_pred = model.predict(X_test)

# Displaying coefficients
print("Coefficients: ", model.coef_)

# Displaying the intercept
print("Intercept: ", model.intercept_)

from sklearn.metrics import r2_score
[r2_score(y_test, y_pred), r2_score(y_train,model.predict(X_train))]

X_test_frame = pd.DataFrame([y_pred,y_test])
X_test_frame = X_test_frame.T
X_test_frame.columns = ['predicted','actual']
X_test_frame.plot.scatter(x='predicted',y='actual')




###############################
# 7 day over under windows
##############################


def calculate_previous_7_days_mean(row, df, stat):
    # Filter the DataFrame for the same season and dates before the current row's date
    filtered_df = df[(df['season'] == row['season']) & (df['date'] < row['date'])]
    
    # Get the last 7 unique dates
    unique_dates = filtered_df['date'].unique()[-7:]
    
    if len(unique_dates) < 7:
        # If there are not 7 unique dates, return 10
        return 10
    else:
        # Calculate the mean of 'over_under_result' for those dates
        mean_value = filtered_df[filtered_df['date'].isin(unique_dates)][stat].mean()
        return mean_value

def calculate_next_6_days_mean(row, df,stat):
    # Filter the DataFrame for dates starting from the current row's date
    filtered_df = df[df['date'] >= row['date']]
    
    # Get the first 7 unique dates after including the current date
    unique_dates = filtered_df['date'].unique()[:7]
    
    if len(unique_dates) < 7:
        # If there are less than 7 dates available, return NaN or some placeholder value
        return np.nan
    else:
        # Calculate the mean of 'over_under_result' for those dates
        mean_value = filtered_df[filtered_df['date'].isin(unique_dates)][stat].mean()
        return mean_value



data['prev_7_days_over_under_avg'] = data.apply(lambda row: calculate_previous_7_days_mean(row, data,stat='over_under_result'), axis=1)
data['prev_7_days_fav_spread_cover_avg'] = data.apply(lambda row: calculate_previous_7_days_mean(row, data,stat='favorite_spread_cover'), axis=1)

data['next_7_days_over_under_avg'] = data.apply(lambda row: calculate_next_6_days_mean(row, data,stat='over_under_result'), axis=1)
data['next_7_days_fav_spread_cover_avg'] = data.apply(lambda row: calculate_next_6_days_mean(row, data,stat='favorite_spread_cover'), axis=1)


subset = data[(data['prev_7_days_over_under_avg'] != 10) & (data['next_7_days_over_under_avg'] != 10)]

subset['prev_7_days_over_under_avg'].hist(bins=50)

mostly_unders = subset[subset['prev_7_days_over_under_avg'] <= 0.4]
mostly_unders['next_7_days_over_under_avg'].describe()

mostly_overs = subset[subset['prev_7_days_over_under_avg'] >= 0.55]
mostly_overs['next_7_days_over_under_avg'].describe()


mostly_favorites = subset[subset['prev_7_days_fav_spread_cover_avg'] >= 0.57]
mostly_favorites['next_7_days_fav_spread_cover_avg'].describe()

mostly_dogs = subset[subset['prev_7_days_fav_spread_cover_avg'] <= 0.45]
mostly_dogs['next_7_days_fav_spread_cover_avg'].describe()

###############################
# LEAGUE ADJUSTED DISTRIBUTIONS
##############################



filtered_data = filter_data(data=data)
stats_to_adjust = ['2pt_percent','3pt_percent','turnover_percentage','off_reb_percentage',
                   'points_per_possession']

for stat in stats_to_adjust:
    for location in ['home','away']:
        stat_medians = filtered_data.groupby('season')[f'12_game_{location}_{stat}'].quantile(0.5)#.loc[2016].values[0]

        # Map the median values to the larger DataFrame
        filtered_data[f'season_{location}_{stat}_median'] = data['season'].map(stat_medians)

        # Calculate the 'over_under_adjusted' column
        filtered_data[f'{location}_{stat}_adjusted'] = filtered_data[f'12_game_{location}_{stat}'] / filtered_data[f'season_{location}_{stat}_median']

adj_cols = [col for col in filtered_data.columns if 'adjust' in col 
            or 'opening' in col or 'total' in col or 'expected_pace' in col or 'points_per_possession' in col or 'train_accuracies' in col
            or 'season' in col]

adj_frame = filtered_data[adj_cols]

for tpile in [0.25,0.5,0.75]:
    for spile in [0.25,0.5,0.75]:
        bottom_total = adj_frame['over_under_adjusted'].quantile(tpile-0.25)
        top_total = adj_frame['over_under_adjusted'].quantile(tpile)

        bottom_spread = adj_frame['opening_spread'].quantile(spile-0.25)
        top_spread = adj_frame['opening_spread'].quantile(spile)

        adj_tspread = adj_frame[(adj_frame['over_under_adjusted'] >= (bottom_total)) &
                          (adj_frame['over_under_adjusted']<= (top_total)) & 
                          (adj_frame['opening_spread'] >= (bottom_spread)) &
                          (adj_frame['opening_spread'] <= (top_spread))]
        

        total_medians = adj_tspread.groupby('season')['opening_ou'].quantile(0.5)#.loc[2016].values[0]
        spread_medians = adj_tspread.groupby('season')['opening_spread'].quantile(0.5)#.loc[2016].values[0]





###############################
# REGRESS OPENER
##############################

filtered_data = filter_data(data)

filtered_data['o/u'].corr(filtered_data['total_points'])
filtered_data['opening_ou'].corr(filtered_data['total_points'])

#spread is positive if home team is favored

filtered_data['team_net_rtg_diff'] = filtered_data['home_net_rtg'] - filtered_data['away_net_rtg']
filtered_data['predicted_spread'] = filtered_data['team_net_rtg_diff'] * filtered_data['12_game_expected_pace']
filtered_data['const'] = 1
filtered_data['rescaled_pace'] = filtered_data['12_game_expected_pace'] / 100
clean_openers = filtered_data.dropna(subset=['opening_spread'])



#X = clean_openers[['team_net_rtg_diff','12_game_expected_pace']]
X = clean_openers[['o/u','12_game_expected_pace','team_net_rtg_diff']]

y = clean_openers['total_points']



# Creating a linear regression model
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the model using the training sets
model.fit(X_train, y_train)

# Making predictions using the testing set
y_pred = model.predict(X_test)

# Displaying coefficients
print("Coefficients: ", model.coef_)

# Displaying the intercept
print("Intercept: ", model.intercept_)

from sklearn.metrics import r2_score
[r2_score(y_test, y_pred), r2_score(y_train,model.predict(X_train))]

X_test_frame = pd.DataFrame([y_pred,y_test])
X_test_frame = X_test_frame.T
X_test_frame.columns = ['predicted','actual']
X_test_frame.plot.scatter(x='predicted',y='actual')





##############################################
 # optimal weights to get pace
##############################################
df = filter_data(data)

pace_df = df[['12_game_home_possessions','12_game_away_possessions','pace']]

X = pace_df[['12_game_home_possessions','12_game_away_possessions']]
y = pace_df['pace']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model using the training sets
model.fit(X_train, y_train)

# Making predictions using the testing set
y_pred = model.predict(X_test)

# Displaying coefficients
print("Coefficients: ", model.coef_)

# Displaying the intercept
print("Intercept: ", model.intercept_)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

 
r2_score

df['pace'].corr(df['total_points'])


##############################################
 # high var vs low var
##############################################

df = filter_data(data)
df = df[df['12_game_home_3pt_percent_stdev'] != 10]
#df = df[(df['spread_difference'] >=)]
df = df[~df.season.isin([2016,2018])]

stat_cols = ['12_game_home_3pt_rate_stdev',
 '12_game_home_2pt_rate_stdev',
 '12_game_home_3pt_percent_stdev',
 '12_game_home_2pt_percent_stdev',
 '12_game_home_turnover_percentage_stdev',
 '12_game_home_off_reb_percentage_stdev',
 '12_game_home_points_per_possession_stdev',
 '12_game_away_3pt_rate_stdev',
 '12_game_away_2pt_rate_stdev',
 '12_game_away_3pt_percent_stdev',
 '12_game_away_2pt_percent_stdev']

#stat variances throughout seasons broadly the same though a few outliers in 3pt percent
#not a correlation btwn spot and vol, but from underlying vols and spot
df.groupby('season')[['12_game_home_points_per_possession_stdev']].quantile(0.2)#.loc[2016].values[0]
df.groupby('season')[['12_game_away_3pt_percent_stdev']].mean()
df['12_game_home_off_reb_percentage_stdev'].corr(df['12_game_home_points_per_possession'])

#generally all p normal distributions
df['12_game_home_turnover_percentage_stdev'].hist(bins=50) 
home_favor = df[df['home_favor'] ==1 ]
home_favor['12_game_home_points_per_possession'].corr(home_favor['spread_difference'])
home_favor.plot.scatter(x='12_game_home_turnover_percentage_stdev',y='spread_difference')

home_favor['12_game_home_2pt_percent'].corr(home_favor['home_2pt_percent'])

home_favor[(home_favor['12_game_home_2pt_percent'] < 0.48) &
           (home_favor['12_game_away_opponent_2pt_percent'] < 0.52)].home_spread_cover.mean()
home_favor[home_favor['12_game_home_points_per_possession_stdev'] > 0.12].home_spread_cover.shape[0]










##############################################
 # seeing efficiency of spread and total line moves
##############################################


opener_review = pd.read_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/2013_2022_opener_review.csv')

opener_review = opener_review[(opener_review['opening_ou'] >= 150) & (opener_review['opening_ou'] <= 275) &
                              (opener_review['spread'] <= 25) ]
#opener_review = opener_review[opener_review['season'] >= 2018]
opener_review['closing_spread_change'] = opener_review['spread'] - opener_review['opening_spread']
opener_review['closing_total_change'] = opener_review['o/u'] - opener_review['opening_ou']

opener_review['total_points'] = opener_review['finalHome'] + opener_review['finalAway']

opener_review['opener_over_under_result'] = ((opener_review['finalHome'] + opener_review['finalAway']) > opener_review['opening_ou']).astype(int)
opener_review['closing_over_under_result'] = ((opener_review['finalHome'] + opener_review['finalAway']) > opener_review['o/u']).astype(int)

opener_review['home_point_diff'] = opener_review['finalHome'] - opener_review['finalAway']
opener_review['away_point_diff'] = opener_review['finalAway'] - opener_review['finalHome']

opener_review.loc[opener_review['homeFavor'] == 1, 'favorite_point_diff']  = opener_review['finalHome'] - opener_review['finalAway']
opener_review.loc[opener_review['awayFavor'] == 1, 'favorite_point_diff']  = opener_review['finalAway'] - opener_review['finalHome']



fav_flip = opener_review[opener_review['favoriteFlip'] == 1]



fav_flip['homeFavor']

spread_diff = 2.5
no_flip = opener_review[opener_review['favoriteFlip'] == 0]

no_flip.loc[(no_flip['spread'] - no_flip['opening_spread']) >= spread_diff, 'more_favorite' ] = 1
no_flip['more_favorite'] = no_flip['more_favorite'].fillna(0)

no_flip.loc[(no_flip['spread'] - no_flip['opening_spread']) <= -1 * spread_diff, 'less_favorite' ] = 1
no_flip['less_favorite'] = no_flip['less_favorite'].fillna(0)

### FAVORITE MORE FAVORED

towards_favorite = no_flip[no_flip['more_favorite'] == 1]
towards_favorite.loc[ towards_favorite['favorite_point_diff'] >= towards_favorite['opening_spread'] ,'opening_cover'] = 1
towards_favorite['opening_cover'] = towards_favorite['opening_cover'].fillna(0)

towards_favorite.loc[ towards_favorite['favorite_point_diff'] >= towards_favorite['spread'] ,'closing_cover'] = 1
towards_favorite['closing_cover'] = towards_favorite['closing_cover'].fillna(0)

[towards_favorite.opening_cover.mean(), towards_favorite.closing_cover.mean(), towards_favorite.shape[0]]

towards_favorite.groupby('season')[['opening_cover','closing_cover']].mean()



### FAVORITE LESS FAVORED BUT STILL FAVORED

towards_underdog = no_flip[no_flip['less_favorite'] == 1]
towards_underdog.loc[ towards_underdog['favorite_point_diff'] >= towards_underdog['opening_spread'] ,'opening_cover'] = 1
towards_underdog['opening_cover'] = towards_underdog['opening_cover'].fillna(0)

towards_underdog.loc[ towards_underdog['favorite_point_diff'] >= towards_underdog['spread'] ,'closing_cover'] = 1
towards_underdog['closing_cover'] = towards_underdog['closing_cover'].fillna(0)
[towards_underdog.opening_cover.mean(), towards_underdog.closing_cover.mean(), towards_underdog.shape[0]]

towards_underdog.groupby('season')[['opening_cover','closing_cover']].mean()



#TOTAL MOVES


min_line_move = 4

opener_review.loc[(opener_review['o/u'] - opener_review['opening_ou']) >= min_line_move, 'total_up'] = 1
opener_review['total_up'] = opener_review['total_up'].fillna(0)

opener_review.loc[(opener_review['o/u'] - opener_review['opening_ou']) <= -1 * min_line_move, 'total_down'] = 1
opener_review['total_down'] = opener_review['total_down'].fillna(0)

totals_up = opener_review[opener_review['total_up'] == 1]
[totals_up['opener_over_under_result'].mean(), totals_up['closing_over_under_result'].mean()]

totals_down = opener_review[opener_review['total_down'] == 1]
[totals_down['opener_over_under_result'].mean(), totals_down['closing_over_under_result'].mean()]

[totals_up.shape[0], totals_down.shape[0]]

totals_up.groupby('season')[['opener_over_under_result','closing_over_under_result']].mean()
totals_down.groupby('season')[['opener_over_under_result','closing_over_under_result']].mean()

##############################################
 # what stats drive outlier performances
##############################################


filtered = filter_data(data)
filtered['total_points'] = filtered['finalAway'] + filtered['finalHome']
filtered['total_difference'] = filtered['total_points'] - filtered['o/u']

filtered['combined_3pt_points_above_ex'] = filtered['5_game_home_3pt_points_above_ex'] + filtered['5_game_away_3pt_points_above_ex']

over_totals = filtered[filtered['total_difference'] >= 15]
under_totals = filtered[filtered['total_difference'] <= -15]

normal = filtered[filtered['total_difference'] < 10]

over_totals['12_result_relative_pace'].hist(bins=50)
normal['12_result_relative_pace'].hist(bins=50)

#pace is 4% higher - what causes higher pace what causes higher points per possession - wwork down the tree

new_betting_cols = ['5_game_home_total_over','10_game_home_total_over', '5_game_away_total_over','10_game_away_total_over','5_game_home_3pt_points_above_ex','5_game_away_3pt_points_above_ex']
stat = ['combined_3pt_points_above_ex']
over_totals[stat].describe()
normal[stat].describe()



filtered.groupby('season')['total_difference'].quantile(0.25)
filtered.groupby('season')['spread_difference'].quantile(0.25)



# COMPARING TEAMS THAT HAVE BEEN MODELED PROPERLY DIFFERENCES
sig_diff = 10
over_diff = filtered[filtered['total_difference'] >= sig_diff].shape[0] / filtered.shape[0]
under_diff = filtered[filtered['total_difference'] <= -sig_diff].shape[0] / filtered.shape[0]
favor_diff = filtered[filtered['spread_difference'] >= sig_diff].shape[0] / filtered.shape[0]
dog_diff = filtered[filtered['spread_difference'] <= -sig_diff].shape[0] / filtered.shape[0]

spreads_modeled_well = filtered[((filtered['10_game_away_spread_cover'] >= 0.4) & (filtered['10_game_away_spread_cover'] <= 0.6)) 
                             & ((filtered['10_game_home_spread_cover'] >= 0.4) & (filtered['10_game_home_spread_cover'] <= 0.6)) 
                             & (filtered['10_game_away_spread_cover'] != 10)]

spreads_modeled_well[spreads_modeled_well['spread_difference'] >= sig_diff].shape[0] / spreads_modeled_well.shape[0]
spreads_modeled_well[spreads_modeled_well['spread_difference'] <= -sig_diff].shape[0] / spreads_modeled_well.shape[0]

totals_modeled_well = filtered[((filtered['10_game_away_total_over'] >= 0.4) & (filtered['10_game_away_total_over'] <= 0.6)) 
                             & ((filtered['10_game_home_total_over'] >= 0.4) & (filtered['10_game_home_total_over'] <= 0.6)) 
                             & (filtered['10_game_away_spread_cover'] != 10)]


totals_modeled_well[totals_modeled_well['total_difference'] >= sig_diff].shape[0] / totals_modeled_well.shape[0]
totals_modeled_well[totals_modeled_well['total_difference'] <= -sig_diff].shape[0] / totals_modeled_well.shape[0]





##############################################
 # visualizing spread difference based on home away
##############################################

data['home_points_diff'] = data['finalHome'] - data['finalAway']
home_rest = data[data['home_rest'] == 0]

##############################################
 # underdogs home/away perf by year
##############################################

data.groupby('season')['favorite_spread_cover'].mean()


##############################################
 # performance by fast pace low pace teams on b2b
##############################################
def calculate_percentiles(group, percentile_value):
    return group.quantile(percentile_value)


high_pace_vals = data.groupby('season')['12_game_home_possessions'].apply(calculate_percentiles, percentile_value=0.75)
high_pace_vals = high_pace_vals.to_dict()
high_pace_stats = {}
for year in rel_years:
    year_frame = data[data['season'] == year]
    high_pace_no_rest = year_frame[(year_frame['12_game_away_possessions'] >= high_pace_vals[year]) &
                                   (year_frame['away_rest'] <= 1)]
    high_pace_stats[year] = [high_pace_no_rest.away_spread_cover.sum() / high_pace_no_rest.shape[0],
                             high_pace_no_rest.over_under_result.sum() / high_pace_no_rest.shape[0],
                             high_pace_no_rest.shape[0]]
    

    



##############################################
 # strong last 5 vs weak last 5
##############################################
data = filter_data(data)

strong_home_trend = data[(data['5_game_home_spread_cover'] >= 0.8) & (data['5_game_away_spread_cover'] <=0.2)]
strong_home_trend.home_spread_cover.sum() / strong_home_trend.shape[0]

strong_away_trend = data[(data['5_game_away_spread_cover'] >= 0.8) & (data['5_game_home_spread_cover'] <=0.2)]
strong_away_trend.home_spread_cover.sum() / strong_away_trend.shape[0]


strong_strong_trend = data[(data['5_game_away_spread_cover'] >= 0.8) & (data['5_game_home_spread_cover'] >=0.8)]
strong_strong_trend.over_under_result.sum() / strong_strong_trend.shape[0]

weak_weak_trend = data[(data['5_game_away_spread_cover'] <= 0.2) & (data['5_game_home_spread_cover'] <=0.2)]
weak_weak_trend.over_under_result.sum() / weak_weak_trend.shape[0]









##############################################
 # double digit favorites home vs road
##############################################

rel_years = data[data['season'].isin([2017,2018,2019, 2020, 2021, 2022,2023])]

home_double_digi = rel_years[(rel_years['home_favor'] == 1) & (rel_years['spread'] >= 10)]
home_double_digi.home_spread_cover.sum() / home_double_digi.shape[0]

away_double_digi = rel_years[(rel_years['away_favor'] == 1) & (rel_years['spread'] >= 10)]
away_double_digi.home_spread_cover.sum() / away_double_digi.shape[0]


##############################################
 # double digit favorites by team strength
##############################################

good_offense_by_year = {}
good_defense_by_year = {}

bad_defense_by_year = {}
bad_offense_by_year = {}

rel_years = [2017,2018,2019, 2020, 2021, 2022,2023]
for year in rel_years:
    offense_performances = data[(data['season'] == year)][['home_points_per_possession','away_points_per_possession']].values
    flattened_perfs = []
    for perf in offense_performances:
        flattened_perfs.append(perf[0])
        flattened_perfs.append(perf[1])
    a = np.array(flattened_perfs)
    good_offense_by_year[year] = np.percentile(a, 60)
    bad_defense_by_year[year] = np.percentile(a,60)

    good_defense_by_year[year] = np.percentile(a,40)
    bad_offense_by_year[year] = np.percentile(a,40)

filtered_data = filter_data(data)

summary_stats = {}

for year in rel_years:
    year_frame = filtered_data[filtered_data['season'] == year]
    good_offense = good_offense_by_year[year]
    good_defense = good_defense_by_year[year]
    
    bad_offense = good_defense
    bad_defense = good_offense

    home_good_o_bad_d = year_frame[(year_frame['12_game_home_points_per_possession'] >= good_offense) &
                                   (year_frame['12_game_away_opponent_points_per_possession'] >= bad_defense)]
    home_high_spread = home_good_o_bad_d[home_good_o_bad_d['spread']>=5]
    home_low_spread = home_good_o_bad_d[home_good_o_bad_d['spread']<=5]
    summary_stats[f'{year}_home'] = {'high_spread_stats':[home_high_spread.home_spread_cover.sum(),
                                                         home_high_spread.over_under_result.sum(),
                                                         home_high_spread.shape[0]],
                                      'low_spread_stats':[home_low_spread.home_spread_cover.sum(),
                                                         home_low_spread.over_under_result.sum(),
                                                         home_low_spread.shape[0]]}


    away_good_o_bad_d = year_frame[(year_frame['12_game_away_points_per_possession'] >= good_offense) &
                                   (year_frame['12_game_home_opponent_points_per_possession'] >= bad_defense)]
    away_high_spread = away_good_o_bad_d[away_good_o_bad_d['spread']>=5]
    away_low_spread = away_good_o_bad_d[away_good_o_bad_d['spread']<=5]
    summary_stats[f'{year}_away'] = {'high_spread_stats':[away_high_spread.away_spread_cover.sum(),
                                                         away_high_spread.over_under_result.sum(),
                                                         away_high_spread.shape[0]],
                                      'low_spread_stats':[away_low_spread.away_spread_cover.sum(),
                                                         away_low_spread.over_under_result.sum(),
                                                         away_low_spread.shape[0]]}




##############################################
  # 3 in 4
##############################################

rel_years = data[data['season'].isin([2019, 2020, 2021, 2022,2023])]

all_dict = {}

for year in rel_years.season.unique():
    y_frame = rel_years[rel_years['season'] == year]
    locate_results = {}
    for locate in ['home','away']:
          locate_frame = y_frame[(y_frame[f'{locate}_2_in_3'] == 1) & (y_frame['o/u'] <= 255.5)]
          locate_results[locate] = [locate_frame.home_spread_cover.sum() / locate_frame.shape[0],
                                    locate_frame.over_under_result.sum() / locate_frame.shape[0],
                                    locate_frame.shape[0]]
    all_dict[year] = locate_results
all_dict

##############################################
  # HOME SCORING MARGIN CORRELATIONS
##############################################

filtered_data = filter_data(data)
subset = filtered_data[filtered_data['season'].isin([2017,2018,2019,2022,2023])]


matchup_diff_columns = [col for col in data.columns if 'matchup_differential' in col]
spread_columns = ['spread', 'o/u','home_spread_cover', 'home_scoring_margin']
#margin_cols = ['home_scoring_margin','home_spread']
raw_matchup_cols = [col for col in data.columns if 'matchup' in col and 'differential' not in col]
sharpe_cols = [col for col in data.columns if '_adv' in col]
ma_columns = [col for col in data.columns if '12_game' in col and 'stdev' not in col]
relative_offense = [col for col in data.columns if 'home_offense_relative' in col]
relative_defense = [col for col in data.columns if 'away_defense_relative' in col]


matchup_spread_frame = subset[matchup_diff_columns + spread_columns + sharpe_cols + ma_columns + relative_offense + relative_defense]

corr_col = 'spread'

rel_feature_frame = matchup_spread_frame.drop(columns=['home_scoring_margin', 'home_spread_cover'])
individual_correlations = matchup_spread_frame.corr()[corr_col].drop(['home_scoring_margin', 'spread', 'home_spread_cover'])

combined_correlations = {}



# Iterate over combinations of features
for i, feature1 in enumerate(rel_feature_frame):
    for j, feature2 in enumerate(rel_feature_frame):
        if i < j:  # To avoid duplicate combinations and self-combination
            additive_feature = feature1 + '_PLUS_' + feature2
            multiplied_feature = feature1 + '_TIMES_' + feature2
            rel_feature_frame[additive_feature] = rel_feature_frame[feature1] + rel_feature_frame[feature2]
            correlation = rel_feature_frame[additive_feature].corr(rel_feature_frame[corr_col])
            combined_correlations[additive_feature] = correlation
            
            rel_feature_frame[multiplied_feature] = rel_feature_frame[feature1] * rel_feature_frame[feature2]
            correlation = rel_feature_frame[multiplied_feature].corr(rel_feature_frame[corr_col])
            combined_correlations[multiplied_feature] = correlation


# Sort combined correlations
sorted_combined_correlations = {k: v for k, v in sorted(combined_correlations.items(), key=lambda item: item[1], reverse=True)}
combined_corrs = pd.Series(sorted_combined_correlations)
sig_corrs = abs(combined_corrs).nlargest(50)


##############################################
  # NET RATING CORRELATIONS TO SPREAD
##############################################



stats = ['points_per_possession','2pt_rate','3pt_rate',
'2pt_percent','3pt_percent','off_reb_percentage','turnover_percentage']

for stat in stats:
    subset[f'offense_net_{stat}'] = subset[f'12_game_home_{stat}'] - subset[f'12_game_away_{stat}']
    subset[f'defense_net_{stat}'] = subset[f'12_game_home_opponent_{stat}'] - subset[f'12_game_away_opponent_{stat}']

net_stats = [stat for stat in subset.columns if 'net' in stat]
net_stats_spread = subset[net_stats+['spread']]
net_stat_corrs = net_stats_spread.corr()['spread']


##############################################
  # PARLAY WORK
##############################################


subset = data[data['season'].isin([2021,2022,2023])]

home_spread_over = subset[(subset['home_spread_cover'] == 1) & (subset['over_under_result'] == 1)]
home_spread_under = subset[(subset['home_spread_cover'] == 1) & (subset['over_under_result'] == 0)]
away_spread_over = subset[(subset['home_spread_cover'] == 0) & (subset['over_under_result'] == 1)]
away_spread_under = subset[(subset['home_spread_cover'] == 0) & (subset['over_under_result'] == 0)]

frames = {'home_spread_over': home_spread_over,
          'home_spread_under':home_spread_under,
          'away_spread_over':away_spread_over,
          'away_spread_under':away_spread_under}

def check_parlay(frame,all):
    return frame.shape[0] / all.shape[0]

result_dict = {}
total_dict = {}

for total in [[0,210],[210.5,220],[220.5,230],[230.5,250]]:
    just_total = subset[(subset['o/u'] >= total[0]) & (subset['o/u'] <= total[1])]
    for spread in [[0,2.5],[3,5.5],[6,8.5],[9,25]]:
        total_sub = just_total[(just_total['spread'] >= spread[0]) & (just_total['spread'] <= spread[1])]
        favorite_spread_over = total_sub[(total_sub['favorite_spread_cover'] == 1) & (total_sub['over_under_result'] == 1)]
        favorite_spread_under = total_sub[(total_sub['favorite_spread_cover'] == 1) & (total_sub['over_under_result'] == 0)]
        underdog_spread_over = total_sub[(total_sub['favorite_spread_cover'] == 0) & (total_sub['over_under_result'] == 1)]
        underdog_spread_under = total_sub[(total_sub['favorite_spread_cover'] == 0) & (total_sub['over_under_result'] == 0)]
        
        frames = {'favorite_spread_over': favorite_spread_over,
                'favorite_spread_under':favorite_spread_under,
                'underdog_spread_over':underdog_spread_over,
                'underdog_spread_under':underdog_spread_under}
        for id, frame in frames.items():
            total_dict[f'{id} {total} + {spread} {total_sub.shape[0]}'] = check_parlay(frame,total_sub)

#locate dict is based off of favorite spread cover being off of home/away
threshold = 0.3
filtered_pairs = {k: v for k, v in total_dict.items() if v > threshold}



##############################################
  # PACE TOTAL ANALYSES
##############################################
outputs = ['pace','awayPoints','over_under_result']
subset = filter_data(data)

subset = subset[(subset['12_game_away_possessions'] > 70) &(subset['12_game_home_possessions'] > 70)]

pace_check = subset[(subset['o/u'] <=  220)
                             & (subset['12_game_expected_pace'] <=  98)]
                          #  & (subset['spread'] <= 4)]

[pace_check.over_under_result.sum() / pace_check.shape[0], pace_check.shape[0],
 pace_check.favorite_spread_cover.sum() / pace_check.shape[0]]

#checking if just good defenses generally create unders

rel_cols = [col for col in subset.columns if 'relative' in col and 'points'  in col]
pace_cols = data.filter(regex='pace').columns

subset = subset[(subset['12_game_expected_pace'] > 70) &(subset['12_game_home_possessions'] > 70)]



pace_check = subset[((subset['12_game_away_opponent_points_per_possession'] >=  1.15)
                             & (subset['12_game_home_opponent_points_per_possession'] >=  1.15))
                            & (subset['12_game_expected_pace'] >= 100)]
                          #  & (subset['away_favor'] == 1)]

[pace_check.over_under_result.sum() / pace_check.shape[0], pace_check.shape[0],
 pace_check.favorite_spread_cover.sum() / pace_check.shape[0]]




##############################################
  # TREND AND MEAN REVERSION - nothing really
##############################################

### SPREADS

### 10 GAME

test = data[(data['10_game_away_spread_cover'] >= 0.7) 
                             & (data['10_game_home_spread_cover'] <= 0.3)
                             & (data['10_game_away_spread_cover'] != 10)]
test.home_spread_cover.mean()

test.groupby('season')[['home_spread_cover']].mean()

test = data[(data['10_game_home_spread_cover'] >= 0.7) 
                             & (data['10_game_away_spread_cover'] <= 0.3)
                             & (data['10_game_home_spread_cover'] != 10)]
test.away_spread_cover.mean()

test.groupby('season')[['away_spread_cover']].mean()

### 5 GAME

test = data[(data['5_game_away_spread_cover'] >= 0.7) 
                             & (data['5_game_home_spread_cover'] <= 0.3)
                             & (data['5_game_away_spread_cover'] != 10)]
test.home_spread_cover.mean()

test.groupby('season')[['home_spread_cover']].mean()

test = data[(data['5_game_home_spread_cover'] >= 0.7) 
                             & (data['5_game_away_spread_cover'] <= 0.3)
                             & (data['5_game_home_spread_cover'] != 10)]
test.away_spread_cover.mean()



### TOTALS
test = data[(data['5_game_away_total_over'] >= 0.7) 
                             & (data['5_game_home_total_over'] <= 0.3)
                             & (data['5_game_away_spread_cover'] != 10)]
test.over_under_result.mean()

test = data[(data['10_game_away_total_over'] >= 0.8) 
                             & (data['10_game_home_total_over'] <= 0.2)
                             & (data['10_game_away_spread_cover'] != 10)]
test.over_under_result.mean()








                           #  & (data['prev_game_away_point_differential'].abs() >= 12)]
                           # & (data['prev_game_home_point_differential'] <= 10)]
check = [(test.underdog_dollars_won.sum() / test.underdog_dollars_risked.sum()),  test.shape[0]]
check


##############################################
  # SEASONALITY
##############################################


stat_formula = {
    '3pt_rate':['3PA','FGA'],
    '2pt_rate':['2pt_attempts','FGA'],
    '3pt_percent':['3P','3PA'],
    '2pt_percent':['2pt_makes','2pt_attempts'],
    'turnover_percentage':['TOV','possessions'],
    'off_reb_percentage':['ORB','FGA'],
    'points_per_possession': ['Points','possessions'],
    'ft_rate':['FT','FTA']

}

filtered_data = filter_data(data)
subset = filtered_data[filtered_data['season'].isin([2017,2018,2019,2022,2023])]
stat = 'over_under_result'
is_stat = False

season_dict = {}
for season in subset.season.unique():
     s_frame = subset[subset['season'] == season]
     s_frame = s_frame.reset_index()
     s_frame['game_bucket'] = s_frame.index // 50
     bucket_avgs = []
     

     for bucket in s_frame.game_bucket.unique():
          buck_frame = s_frame[s_frame['game_bucket'] == bucket]
          if is_stat:
            home_vals = list(buck_frame[f'home_{stat}'].values)
            home_vals = [val for val in home_vals if val >= 0]
            away_vals = list(buck_frame[f'home_{stat}'].values)
            away_vals = [val for val in away_vals if val >= 0]
            both = home_vals + away_vals
            both = [val for val in both if val <= 1.5]
            bucket_avgs.append(sum(both)/ len(both))
          else:
              vals = buck_frame[stat].values
              bucket_avgs.append(sum(vals) / len(vals))

     season_dict[season] = bucket_avgs

full_length = len(season_dict[2017])

season_dict = {key: value[:full_length - 1] for key, value in season_dict.items()}

season = 2017; bucket = 10

season_frame = pd.DataFrame(season_dict)
season_frame['row_avg'] = season_frame.mean(1)
season_frame['ma'] = season_frame['row_avg'].rolling(5).mean()
season_frame['row_avg'].plot(kind='line')

season_frame[2022].plot(kind='line')
season_frame[2023].plot(kind='line')
season_frame[2019].plot(kind='line')







year_results = {}
month_results = {}
year_month_dict = {}
rel_years = data[data['season'].isin([2019, 2020, 2021, 2022,2023])]

year_day_dict = {}
for month in rel_years.month.unique():
    subset = rel_years[rel_years['month'] == month]
    day_dict = {}
    for day in rel_years.day_of_week.unique():
        day_frame = subset[subset['day_of_week'] == day]
        day_dict[day] = day_frame.favorite_spread_cover.sum() / day_frame.favorite_spread_cover.shape[0]
    year_day_dict[month] = day_dict


## ml dollars

for year in rel_years.season.unique():
    subset = rel_years[rel_years['season'] == year]
    year_results[year] = subset['favorite_dollars_won'].sum() / subset['favorite_dollars_risked'].sum() 
    for month in subset.month.unique():
        
        x = subset[subset['month'] == month]
        year_month_dict[f'{str(month)} / {str(year)}'] = x['favorite_dollars_won'].sum() / x['favorite_dollars_risked'].sum() 
        month_results[month] =  x['favorite_dollars_won'].sum() / x['favorite_dollars_risked'].sum() 

spread_buckets = [[0,4],[4.5,7],[7.5,25]]
total_buckets = [[0,215],[215.5,224],[224.5,232],[232.5,250]]
for bucket in total_buckets:
    s_bucket = rel_years[(rel_years['o/u'] >= bucket[0]) & (rel_years['o/u'] <= bucket[1])]
    for month in rel_years.month.unique():
        if month in [5,7,8]:
            continue
        subset = s_bucket[s_bucket['month'] == month]
        year_results[f'{month} {bucket} {subset.shape[0]}'] = subset['over_under_result'].sum() / subset.shape[0]

"""        for month in subset.month.unique():
            x = subset[subset['month'] == month]
            year_month_dict[f'{str(month)} / {str(year)} {bucket}'] = x['favorite_spread_cover'].sum() / x.shape[0]
            month_results[month] =  x['favorite_spread_cover'].sum()  / x.shape[0]
"""
dec_data= {(k,v) for k,v in year_month_dict.items() if '4 /' in k}
dec_data



for year in data.season.unique():
    subset = data[data['season'] == year]
    for month in subset.month.unique():
        x = subset[subset['month'] == month]
        print(f'{month} / {year}: {x.home_spread_cover.sum() / x.shape[0]}')



### MONEYLINES DIFFERENT B2B DYNAMICS


subset = subset[subset['favorite_dollars_won'].abs() <= 2000]

test = subset[(subset['away_favor'] == 1)
                             & (subset['away_rest'] == 1)]
                           #  & (data['prev_game_away_point_differential'].abs() >= 12)]
                           # & (data['prev_game_home_point_differential'] <= 10)]
check = [(test.underdog_dollars_won.sum() / test.underdog_dollars_risked.sum()),  test.shape[0]]
check

check = [(test.favorite_dollars_won.sum() / test.favorite_dollars_risked.sum()),  test.shape[0]]
check


### UNDERDOGS ATS WITH TIGHT SPREADS BY YEAR
u_spread_cover = {}
for year in sorted(data.season.unique()):
    x = data[(data['spread'] >= 10) & (data['season'] == year)]
    u_spread_cover[year]  = x.underdog_spread_cover.sum() / x.shape[0]


example_col_names = ['5_game_home_opponent_ft_points_above_ex' , '5_game_away_3pt_points_above_ex' , 
'season_away_spread_cover_percent', 'season_away_win_percent', 
'10_game_home_total_over', '10_game_away_spread_cover', 'prev_game_away_pace']


##############################################
  # TEAMS PLAYING EACH OTHER ON B2B BACK
############################################## 

test = data[(data['prev_game_home_opponent'] == data['awayTeam'])
                             & (data['prev_game_away_opponent'] == data['homeTeam'])
                          #   & (data['prev_game_away_point_differential'].abs() >= 12)]
                            & (data['prev_game_home_point_differential'] <= -15)]
check = [(test.home_spread_cover.sum() / test.shape[0]), 
         (test.over_under_result.sum() / test.shape[0]), test.shape[0]]
check

test = subset[(subset['prev_game_home_opponent'] == subset['awayTeam'])
                             & (subset['prev_game_away_opponent'] == subset['homeTeam'])
                             & (subset['prev_game_away_point_differential'].abs() >= 15)]
                           # & (data['prev_game_home_point_differential'] <= 10)]
check = [(test.home_spread_cover.sum() / test.shape[0]), 
         (test.over_under_result.sum() / test.shape[0]), test.shape[0]]
check


##############################################
  # BASIC STRAT TESTS
##############################################


test = data[((data['rolling_home_offense_relative_2pt_percent'] <= 0.95)
                             & (data['rolling_away_defense_relative_2pt_percent'] <= 0.95))]
                            #& (data['home_rest'] == 1)]
1- (test.home_spread_cover.sum() / test.shape[0])



bad_tov_away = data[((data['12_game_away_2pt_rate'] >=  0.575)
                             & (data['rolling_home_defense_relative_2pt_percent'] <=  0.925))
                            & (data['away_rest'] == 1)]
(bad_tov_away.home_spread_cover.sum() / bad_tov_away.shape[0])





### BAD 2PT OFFENSE, GOOD 2PT DEFENSE, SOMEHOW BACKWARDS INTUITION WORKS?

bad_tov_home = data[((data['rolling_home_offense_relative_2pt_percent'] <= 0.95)
                             & (data['rolling_away_defense_relative_2pt_percent'] <= 0.95))
                            & (data['home_rest'] == 1)]
(bad_tov_home.home_spread_cover.sum() / bad_tov_home.shape[0])



bad_tov_away = data[((data['rolling_away_offense_relative_2pt_percent'] <=  0.95)
                             & (data['rolling_home_defense_relative_2pt_percent'] <=  0.95))
                            & (data['away_rest'] == 1)]
1- (bad_tov_away.home_spread_cover.sum() / bad_tov_away.shape[0])






##############################################
  # BASIC STRAT TESTS
##############################################

subset = data[data['season'] == 2023]



stats = ['points_per_possession','2pt_rate','3pt_rate',
'2pt_percent','3pt_percent','off_reb_percentage','turnover_percentage']

home_results = []

away_results = []

for stat in stats:
    types = ['12_game_','relative','sharpe']
    for type in types:
        if type == '12_game_':
            home_offense_col = f'12_game_home_{stat}'
            home_defense_col = f'12_game_home_opponent_{stat}'
            away_offense_col = f'12_game_away_{stat}'
            away_defense_col = f'12_game_away_opponent_{stat}'
        elif type == 'relative':
            home_offense_col = f'rolling_home_offense_relative_{stat}'
            home_defense_col = f'rolling_home_defense_relative_{stat}'
            away_offense_col = f'rolling_away_offense_relative_{stat}'
            away_defense_col = f'rolling_away_defense_relative_{stat}'
        elif type == 'sharpe':
            home_offense_col = f'home_{stat}_sharpe'
            home_defense_col =  f'home_{stat}_opponent_sharpe'
            away_offense_col =  f'away_{stat}_sharpe'
            away_defense_col = f'away_{stat}_opponent_sharpe'

        good_offense = subset[home_offense_col].quantile(0.2)
        bad_defense = subset[home_defense_col].quantile(0.2)

        if stat == 'turnover_percentage':
            good_offense = subset[home_offense_col].quantile(0.8)
            good_defense = subset[home_offense_col].quantile(0.8)

            home_test = subset[((subset[home_offense_col] >= good_offense)
                                & (subset[away_defense_col] >= bad_defense))]
            away_test = subset[((subset[away_offense_col] <= good_offense)
                                & (subset[home_defense_col] <= bad_defense))]
        else:
            home_test = subset[((subset[home_offense_col] <= good_offense)
                                    & (subset[away_defense_col] <= bad_defense))]
            
            away_test = subset[((subset[away_offense_col] <= good_offense)
                                    & (subset[home_defense_col] <= bad_defense))]


        home_results.append([stat,type, home_test.home_spread_cover.sum() / home_test.shape[0], home_test.shape[0] ] )

        away_results.append( [stat,type, 1 - (away_test.home_spread_cover.sum() / away_test.shape[0]), away_test.shape[0] ] )

home_offense_good = pd.DataFrame(home_results,columns=['stat','type','cover_percent','count'])
away_offense_good = pd.DataFrame(away_results,columns=['stat','type','cover_percent','count'])







##############################################
  # BAD STRATEGIES
##############################################

# hot three point teams

fade_hot_three_home = data[((data['12_game_home_3pt_rate'] >= 0.4)
                             & (data['12_game_home_3pt_percent'] >= 0.375))
                             &  ((data['12_game_away_opponent_3pt_rate'] <= 0.35))]
                         #    & (data['home_rest'] == 1)]
1- (fade_hot_three_home.home_spread_cover.sum() / fade_hot_three_home.shape[0])

fade_hot_three_away = data[((data['12_game_away_3pt_rate'] >= 0.4)
                             & (data['12_game_away_3pt_percent'] >= 0.375))
                             & ((data['12_game_home_opponent_3pt_rate'] <= 0.35))]
(fade_hot_three_away.home_spread_cover.sum() / fade_hot_three_away.shape[0])


# high tov vs high allowing tov team - MEH RESULTS away hits 63% on 41 games


bad_tov_home = data[((data['12_game_home_turnover_percentage'] >= 0.15)
                             & (data['12_game_away_opponent_turnover_percentage'] >= 0.15))
                            & (data['home_rest'] == 1)]
1- (bad_tov_home.home_spread_cover.sum() / bad_tov_home.shape[0])

bad_tov_away = data[((data['12_game_away_turnover_percentage'] >= 0.155)
                             & (data['12_game_home_opponent_turnover_percentage'] >= 0.155))]
                          #  & (data['away_rest'] == 1)]
(bad_tov_away.home_spread_cover.sum() / bad_tov_away.shape[0])

# low tov vs high tov allowing team - nothing

bad_tov_home = data[((data['12_game_home_turnover_percentage'] <= 0.115)
                             & (data['12_game_away_opponent_turnover_percentage'] <= 0.115))]
                           # & (data['home_rest'] == 1)]
(bad_tov_home.home_spread_cover.sum() / bad_tov_home.shape[0])

bad_tov_away = data[((data['12_game_away_turnover_percentage'] <= 0.115)
                             & (data['12_game_home_opponent_turnover_percentage'] <= 0.115))]
                           # & (data['away_rest'] == 1)]
1- (bad_tov_away.home_spread_cover.sum() / bad_tov_away.shape[0])



