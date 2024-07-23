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
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial import distance
from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors
import numpy as np




new_file =pd.read_csv('')


data = pd.read_csv('')


data['datetime_date'] = pd.to_datetime(data['finalDate'])
data['month'] = data['datetime_date'].dt.month
data = data[data['regSeason'] == 1]
data['favorite_dollars_risked'] = data['favorite_moneyline'].abs()
data['underdog_dollars_risked'] = 100

data['home_point_differential'] = data['finalHome'] - data['finalAway']
data['away_point_differential'] = data['finalAway'] - data['finalHome']

total_medians = data.groupby('season')[['o/u']].quantile(0.5)#.loc[2016].values[0]

data['season_med_ou'] = data['season'].map(total_medians['o/u'])

data['over_under_adjusted'] = data['o/u'] / data['season_med_ou']



def get_feature_weights(feat_series, n_largest=12):
    feature_names = feat_series.nlargest(n_largest).index
    rf_features = feat_series.nlargest(n_largest).values
    return [val / rf_features.sum() for val in rf_features], feature_names


def average_value_for_neighbors(data, column, neighbors):
    neighbor_values = data.iloc[[idx for idx, _ in neighbors]][column]
    return neighbor_values.mean()

# Function to find weighted K nearest neighbors
def weighted_knn(data, query, weights, k=5):
    distances = []
    for idx, row in data.iterrows():
        weighted_dist = distance.euclidean(row * weights, query * weights)
        distances.append((idx, weighted_dist))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors


def run_knn(all_data, scaled_data,weights,neighbor_count,column_to_average,
            main_frame = filtered_data):
    averages = []
    neighbor_stats = {}

    for idx,_ in scaled_data.iterrows():
        neighbors = weighted_knn(scaled_data, scaled_data.loc[idx].values, weights, k=neighbor_count)[1:]
        avg_value = average_value_for_neighbors(all_data, column_to_average, neighbors)
        averages.append(avg_value)
        
        #getting spreads, totals and other distributions
        neighbor_idxs = [i[0] for i in neighbors]
        #neighbor_idxs = [neighbor_idxs]
        #train_data.loc[idx, 'neighbor_idxs'] =neighbor_idxs

        neighbor_spreads = main_frame.loc[neighbor_idxs]['spread'].values
        train_data.loc[idx,'neighbor_spreads_25'] = np.percentile(neighbor_spreads,0.25)
        train_data.loc[idx,'neighbor_spreads_50'] = np.percentile(neighbor_spreads,0.5)
        train_data.loc[idx,'neighbor_spreads_75'] = np.percentile(neighbor_spreads,0.75)

        neighbor_totals = main_frame.loc[neighbor_idxs]['o/u'].values
        train_data.loc[idx,'neighbor_totals_25'] = np.percentile(neighbor_totals,0.25)
        train_data.loc[idx,'neighbor_totals_50'] = np.percentile(neighbor_totals,0.5)
        train_data.loc[idx,'neighbor_totals_75'] = np.percentile(neighbor_totals,0.75)



        neighbor_stats[idx] = [neighbor_idxs,neighbor_spreads,neighbor_totals]

    train_data['neighbor_spread_cover'] = averages

    return train_data, neighbor_stats


def find_nearest_neighbors(data, query, weights, k=5):
    
    weighted_data = data * weights
    weighted_query = query * weights
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(weighted_data)
    distances, indices = nbrs.kneighbors([weighted_query])
    
    return indices[0]


def run_knn_optimized(all_data, scaled_data, weights, neighbor_count, column_to_average,
                      main_frame=filtered_data):
    if 'over' in column_to_average:
        bet = 'o/u'
    else:
        bet= 'spread'

    scaled_data_np = scaled_data.values
    target_vals = all_data[column_to_average].values
    
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=neighbor_count+1, algorithm='auto').fit(scaled_data_np * weights)
    
    averages = []
    neighbor_stats = {}
    
    for idx, query in enumerate(scaled_data_np):
        distances, indices = nbrs.kneighbors([query * weights])
        neighbors = indices[0][1:] 
        
        avg_value = np.mean(target_vals[neighbors])
        averages.append(avg_value)
       
        neighbor_spreads = main_frame.iloc[neighbors]['spread'].values
        neighbor_totals = main_frame.iloc[neighbors]['o/u'].values
        
        neighbor_stats[idx] = {
            'indexes': neighbors.tolist(),
            'spreads': {
                '25': np.percentile(neighbor_spreads, 25),
                '50': np.percentile(neighbor_spreads, 50),
                '75': np.percentile(neighbor_spreads, 75),
            },
            'totals': {
                '25': np.percentile(neighbor_totals, 25),
                '50': np.percentile(neighbor_totals, 50),
                '75': np.percentile(neighbor_totals, 75),
            }
        }
    
    # Convert averages to a DataFrame column if needed
    all_data[f'neighbor_{bet}_result'] = averages
    
    return all_data, neighbor_stats


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


filtered_data = filter_data(data)
train_data = filtered_data[(filtered_data['season'] != 2023) & (filtered_data['season'] >=2018)]
test_data = filtered_data[filtered_data['season'] == 2023]

matchup_diff_columns = [col for col in train_data.columns if 'matchup_differential' in col]
spread_columns = ['over_under_result']#,'spread', 'o/u', 'home_scoring_margin']
#margin_cols = ['home_scoring_margin','home_spread']
ppp_cols = ['home_points_per_possession','away_points_per_possession']
raw_points_cols = ['finalHome','finalAway']
raw_matchup_cols = [col for col in train_data.columns if 'matchup' in col and 'differential' not in col]
raw_ma_cols = [col for col in train_data.columns if '12_game' in col and 'stdev' not in col]
sharpe_cols = [col for col in train_data.columns if '_adv' in col]
expected_pace_cols = [col for col in train_data.columns if 'expected_pace' in col]


###############################
# home vs away
##############################

custom_cols = ['12_game_home_points_per_possession','12_game_home_opponent_points_per_possession',
               '12_game_away_points_per_possession','12_game_away_opponent_points_per_possession']


X_train = train_data[custom_cols]
X_test = test_data[custom_cols]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled_frame = pd.DataFrame(X_scaled,columns = X_train.columns,index=X_train.index)

X_test_scaled = scaler.transform(X_test)
X_test_scaled_frame = pd.DataFrame(X_test_scaled,columns = X_test.columns,index=X_test.index)


feature_weights = [(1 /len(custom_cols) ) for col in custom_cols]


train_results, train_neighbor_stats = run_knn_optimized(all_data=train_data, scaled_data=X_scaled_frame,
                                     weights=feature_weights, neighbor_count=50,
                                      column_to_average='over_under_result' )


high_under = train_results[train_results['neighbor_o/u_result'] <= 0.4]
high_over =  train_data[train_data['neighbor_o/u_result'] >= 0.6]

high_under.over_under_result.mean() 
high_over.over_under_result.mean()

###############################
# home offense away defense
##############################


ma_frame = train_data[raw_ma_cols + raw_points_cols + spread_columns + expected_pace_cols]

home_o_cols = [col for col in ma_frame.columns if ('12_game_home' in col) & ('opponent' not in col)]


home_o_stats = ma_frame[home_o_cols]


# Assuming X is your features dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(home_o_stats)

pca = PCA(n_components=3)  # 'mle' can automatically choose the number of components or set it to a fixed number
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)

#pca = PCA(n_components=0.95) # Retain 95% of the variance

# Get the PCA components (loadings)
components = pca.components_

# Create a DataFrame from the PCA components
components_df = pd.DataFrame(components, columns=home_o_stats.columns, index=[f'PC{i+1}' for i in range(components.shape[0])])

# Absolute values of component loadings to understand the magnitude of influence
components_abs = np.abs(components_df)

# Sorting components by their influence on each principal component
sorted_components = components_abs.apply(lambda x: x.sort_values(ascending=False).index)

print("Top contributing features for each principal component:")
print(sorted_components)


home_o_stats = pca.fit_transform(home_o_stats)

away_d_cols = [col for col in ma_frame.columns if ('12_game_away_opponent' in col)]
home_review = ma_frame[home_o_cols + away_d_cols + ['finalHome','over_under_result']]
#home_review = ma_frame[home_o_cols + away_d_cols + expected_pace_cols+ ['finalHome','over_under_result']]

corr_matrix = ma_frame[away_d_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()

# Separate features and target
X = home_review.drop(['finalHome','over_under_result'], axis=1)
y = home_review['finalHome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_frame = pd.DataFrame(X_scaled,columns = X.columns,index=X.index)

model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# Extract feature importance from Random Forest
rf_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
#rf_features = rf_importance[rf_importance > 0.03].index
rf_features = rf_importance.nlargest(12).values


feature_weights, feature_names = get_feature_weights(feat_series=rf_importance,n_largest=12)
X_scaled_frame = X_scaled_frame[feature_names]
X_scaled = X_scaled_frame.values

# Add the averages to the original dataframe

home_train_data, home_train_neighbor_stats = run_knn_optimized(all_data=home_review, scaled_data=X_scaled_frame,
                                     weights=feature_weights, neighbor_count=50,
                                      column_to_average='over_under_result' )



"""
# Update DataFrame with new columns
for index, data in home_train_neighbor_stats.items():
    home_train_data.at[index, 'indexes'] = str(data['indexes'])  # Convert list to string to store in DataFrame
    for key in data['spreads']:
        home_train_data.at[index, f'spreads_{key}'] = data['spreads'][key]
    for key in data['totals']:
        home_train_data.at[index, f'totals_{key}'] = data['totals'][key]
"""



test_ma_frame = test_data[home_o_cols + away_d_cols + ['home_points_per_possession','home_spread_cover']]

# Separate features and target
X_test = test_ma_frame.drop(['home_points_per_possession','home_spread_cover'], axis=1)


home_test_data, home_test_neighbor_stats =  run_knn(all_data=home_review, scaled_data=X_test_scaled,
                                     weights=feature_weights, neighbor_count=50,
                                      column_to_average='over_under_result' )



##############################
#away offense vs home defense
##############################
ma_frame = train_data[raw_ma_cols + ppp_cols + ['away_points_per_possession','home_spread_cover']]
away_o_cols = [col for col in ma_frame.columns if ('12_game_away' in col) & ('opponent' not in col)]
home_d_cols = [col for col in ma_frame.columns if ('12_game_home_opponent' in col)]
away_review = ma_frame[away_o_cols + home_d_cols + ['away_points_per_possession','home_spread_cover']]



# Separate features and target
X = away_review.drop(['away_points_per_possession','home_spread_cover'], axis=1)
y = away_review['away_points_per_possession']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# Extract feature importance from Random Forest
rf_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_features = rf_importance[rf_importance > 0.03].index

# Use feature importances as weights
feature_weights = model.feature_importances_

test_ma_frame = test_data[away_o_cols + home_d_cols + ['away_points_per_possession','over_under_result']]

# Separate features and target
X_test = test_ma_frame.drop(['away_points_per_possession','home_spread_cover'], axis=1)

# Standardize features
X_test_scaled = scaler.transform(X_test)


away_train_data, home_train_neighbor_stats = run_knn(all_data=away_review, scaled_data=X_scaled_frame,
                                     weights=feature_weights, neighbor_count=50,
                                      column_to_average='over_under_result' )

test_ma_frame = test_data[home_o_cols + away_d_cols + ['home_points_per_possession','home_spread_cover']]

# Separate features and target
X_test = test_ma_frame.drop(['home_points_per_possession','home_spread_cover'], axis=1)


away_test_data, away_test_neighbor_stats =  run_knn(all_data=away_review, scaled_data=X_test_scaled,
                                     weights=feature_weights, neighbor_count=50,
                                      column_to_average='over_under_result' )

#########################
# Data Review
#########################

#### TOTALS

high_home = home_train_data[home_train_data['neighbor_o/u_result'] <= 0.5]
high_away =  train_data[train_data['away_neighbor_spread_cover'] >= 0.6]

high_home_low_away = train_data[(train_data['home_neighbor_spread_cover'] >= 0.6) 
                          &(train_data['away_neighbor_spread_cover'] <= 0.4)]
low_home_high_away = train_data[(train_data['home_neighbor_spread_cover'] <= 0.4) 
                          &(train_data['away_neighbor_spread_cover'] >= 0.6)]

high_home.over_under_result.mean() 
high_away.over_under_result.mean()
high_home_low_away.over_under_result.mean()
low_home_high_away.over_under_result.mean()



#### SPREADS

high_home = test_data[test_data['home_neighbor_spread_cover'] >= 0.56]
high_away =  test_data[test_data['away_neighbor_spread_cover'] >= 0.56]

high_home_low_away = test_data[(test_data['home_neighbor_spread_cover'] >= 0.56) 
                          &(test_data['away_neighbor_spread_cover'] <= 0.44)]
low_home_high_away = test_data[(test_data['home_neighbor_spread_cover'] <= 0.44) 
                          &(test_data['away_neighbor_spread_cover'] >= 0.56)]


