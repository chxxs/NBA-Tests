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
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
#from sklearn.externals import joblib 

##############################################
  # DATA CLEANING AND LOADING
##############################################


data = pd.read_csv()


data['spread_change'] = abs(data['spread'] - data['opening_spread'])

data.loc[(data['home_spread_cover'] == 0) & (data['spread_difference'] < 0), 'away_spread_cover'] = 1

data.loc[data.home_favor == 1, 'home_spread'] = data['spread']
data.loc[data.home_favor == 0, 'home_spread'] = -data['spread']



#IF WANT TO CONSIDER THIS AN ADJUSTMENT FOR HEALTH
data = data[data['spread_change'] <= 3]




nba_data_filtered = data[(data['regSeason'] == 1) & (data['season'] >= 2016) & (data['season'] != 2021)]


nba_data_filtered = nba_data_filtered[(nba_data_filtered['12_game_home_3pt_rate'] != 10) &
                                      (nba_data_filtered['12_game_away_3pt_rate'] != 10) &
                                      (nba_data_filtered['rolling_home_defense_relative_3pt_rate'] < 1.5) &
                                      (nba_data_filtered['rolling_away_defense_relative_3pt_rate'] <1.5) &
                                      (nba_data_filtered['away_total_shot_attempts'] > 60) &
                                      (nba_data_filtered['home_total_shot_attempts'] > 60) & 
                                      (nba_data_filtered['spread_difference'] != 0) &
                                      (nba_data_filtered['spread'] <= 20) &
                                      (nba_data_filtered['o/u'] <= 275) &
                                      (nba_data_filtered['o/u'] >= 150) & 
                                      (data['home_points_per_possession'] <= 1.75) &
                                      (data['away_points_per_possession'] <= 1.75)]

nba_data_filtered = nba_data_filtered.replace([np.inf, -np.inf], np.nan).dropna()

nba_data_filtered.reset_index(drop=True, inplace=True)
nba_data_filtered.index = nba_data_filtered.index.to_series().astype('int64')

#small multi class function if want to predict spread difference
"""

def multi_class(value):
    if value <= -10:
        return 0
    elif -9 <= value < 0:
        return 1
    elif 1 <= value <= 9:
        return 3
    elif value > 10:
        return 4
    else:
        return 2  # for value == 0


nba_data_filtered['multi_class_spread'] = nba_data_filtered['spread_difference'].apply(multi_class)


"""


stats_to_adjust = ['2pt_percent','3pt_percent','turnover_percentage','off_reb_percentage',
                   'points_per_possession']


#ADJUSTING STATS TO MEDIAN OF THAT YEAR
for stat in stats_to_adjust:
    for location in ['home','away']:
        stat_medians = nba_data_filtered.groupby('season')[f'12_game_{location}_{stat}'].quantile(0.5)#.loc[2016].values[0]

        # Map the median values to the larger DataFrame
        nba_data_filtered[f'season_{location}_{stat}_median'] = data['season'].map(stat_medians)

        # Calculate the 'over_under_adjusted' column
        nba_data_filtered[f'{location}_{stat}_adjusted'] = nba_data_filtered[f'12_game_{location}_{stat}'] / nba_data_filtered[f'season_{location}_{stat}_median']

#PROJECTING STATS BASED ON O V D
for stat in stats_to_adjust:

    nba_data_filtered[f'home_{stat}_proj'] = (data[f'12_game_home_{stat}'] / data[f'league_average_{stat}']) - (1-((data[f'12_game_away_opponent_{stat}'] / data[f'league_average_{stat}']))) * data[f'league_average_{stat}']
    nba_data_filtered[f'away_{stat}_proj'] = (data[f'12_game_away_{stat}'] / data[f'league_average_{stat}']) - (1-((data[f'12_game_home_opponent_{stat}'] / data[f'league_average_{stat}']))) * data[f'league_average_{stat}']



league_average_columns = [col for col in nba_data_filtered.columns if 'league_average' in col or 'avg' in col.lower()]
nba_data_filtered = nba_data_filtered.drop(columns=league_average_columns)
nba_data_filtered = nba_data_filtered[(nba_data_filtered['away_points_per_possession_proj'] >= 0.5) & (nba_data_filtered['away_points_per_possession_proj'] <= 2)]


proj_cols = [col for col in nba_data_filtered.columns if 'proj' in col]
adj_cols = [col for col in nba_data_filtered.columns if 'adjust' in col]
ma_columns = [col for col in nba_data_filtered.columns if '12_game' in col and 'stdev' not in col]
matchup_diff_columns = [col for col in nba_data_filtered.columns if 'matchup_differential' in col]
spread_columns = ['spread','o/u']
spread_result_cols = ['home_spread_cover' ]#'multi_class_spread'] #, 'home_scoring_margin']
total_result_cols = ['over_under_result']
margin_cols = ['home_spread']
raw_matchup_cols = [col for col in nba_data_filtered.columns if 'matchup' in col and 'differential' not in col]
sharpe_cols = [col for col in nba_data_filtered.columns if '_adv' in col and 'home_rest' not in col]
new_betting_cols = ['5_game_home_total_over','10_game_home_total_over','5_game_home_3pt_points_above_ex','5_game_away_3pt_points_above_ex']
pace_cols = ['12_game_expected_pace','6_game_expected_pace']
matchup_spread_frame = nba_data_filtered[proj_cols + margin_cols + spread_result_cols
                                             ]


#sns.pairplot(matchup_spread_frame)

#matchup_spread_frame.hist(figsize=(12, 10), bins=30)
#plt.tight_layout()  # Adjusts subplots for better layout
#plt.show()


#matchup_spread_frame.to_csv('')
 ### checking if multiple filters by percentiles lead to results

"""
vals = [0.2,0.4,0.6,0.8,1]
intervals = {}

class_features = ['home_sharpe_2pt_percent_vs_adv','spread','home_sharpe_3pt_percent_vs_adv',
                  'game_off_reb_percentage_matchup_differential','spread','home_rest_adv']
back = 0
big_dict = {}
for f1 in class_features:
    little_dict = {}
    for val in vals:
        more_vals  = [0.2,0.4,0.6,0.8,1]
        bottom = matchup_data_subset[f1].quantile(back)
        top = matchup_data_subset[f1].quantile(val)
        check = matchup_data_subset[(matchup_data_subset[f1] >= bottom) 
                                    & (matchup_data_subset[f1] <= top )]
        for val in more_vals:
            other_bottom
         little_dict[f'{f2}'] = [check.home_spread_cover.sum() / check.shape[0] , check.shape[0]]
        back += 0.2
    big_dict[f1] = little_dict

"""

class_col = 'home_spread_cover'

matchup_data_subset = matchup_spread_frame.copy()

X_subset = matchup_data_subset.drop(columns=[class_col])
y_subset = matchup_data_subset[class_col]

#binning
"""for column in X_subset.columns:
    col_name = f'binned_{column}'
    X_subset[col_name] = pd.qcut(X_subset[column], q=10, labels=False)
    print(f'{column} done')

binned_cols = [col for col in X_subset.columns if 'binned' in col]
X_subset = X_subset[binned_cols]"""

#y_subset = matchup_data_subset['home_scoring_margin']

X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.25, random_state=24)

# Standardizing the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


### DONT NEED THIS CHUNK ANYMORE
"""
# Converting standardized data back to a dataframe
X_standardized_df_subset = pd.DataFrame(X_subset, columns=X_subset.columns)

# Getting the corresponding y values
y_subset_clean = y_subset[X_standardized_df_subset.index]

# Standardizing the cleaned input features
X_standardized_subset_clean = X_standardized_df_subset
X_standardized_subset_clean = scaler.fit_transform(X_standardized_df_subset)

# Converting standardized data back to a dataframe
X_standardized_df_subset_clean = pd.DataFrame(X_standardized_subset_clean, columns=X_standardized_df_subset.columns)
"""
#pca_12 = PCA(n_components=10)
#X_pca_12 = pca_12.fit_transform(X_standardized_df_subset_clean)

#for determining optimal number of components
"""
pca_12 = PCA(n_components=10)
X_pca_12 = pca_12.fit_transform(X_standardized_df_subset_clean)

pca = PCA()
feats = X_subset
pca.fit(feats)

# Calculate explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.tight_layout()

# Cumulative explained variance plot
plt.figure(figsize=(10, 6))
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid',label='Cumulative Explained Variance')
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.tight_layout()

plt.show()"""





##############################################
  # PREPROCESSING/FEATURE ENGINEERING
##############################################

#squared_frame = (X_standardized_df_subset_clean.copy() * 10) ** 2
#X_standardized_df_subset_clean = (X_standardized_df_subset_clean.copy() * 10) ** 2

from scipy import stats

train_frame = pd.DataFrame(X_train, columns=X_subset.columns)
train_frame['y'] = y_train.values
corrs = train_frame.corrwith(train_frame['y'])
rel_corrs = corrs[abs(corrs) > 0.075]
rel_corrs = rel_corrs[rel_corrs.index != 'y']

#corrs = train_frame.corrwith(train_frame['y'].astype('float') , method= stats.pointbiserialr)
"""col_corrs = {}
for col in train_frame.columns:
    if col == '12_game_expected_pace':
        continue
    col_vals = train_frame[col].values
    col_corrs[col] = stats.pointbiserialr(col_vals, y_train).statistic

col_corrs = pd.Series(col_corrs)
rel_corrs = abs(col_corrs).nlargest(20)
"""

selector = SelectKBest(score_func=mutual_info_classif)
selector.fit(X_train, y_train)

# Getting the scores and p-values
scores = selector.scores_
p_values = selector.pvalues_

# Creating a dictionary to store feature names along with their respective scores and p-values
#feature_selection_dict = dict(zip(X_subset.columns, zip(scores, p_values)))
feature_selection_dict = dict(zip(X_subset.columns, scores))
feature_series = pd.Series(feature_selection_dict)
rel_features = abs(feature_series).nlargest(25)

# Sorting the features based on scores in descending order
#sorted_feature_selection_dict = sorted(feature_selection_dict.items(), key=lambda x: x[1][0], reverse=True)
#features = [k for k, v in feature_selection_dict.items() if v > 0.10]

class_features = ['home_sharpe_2pt_percent_vs_adv','spread','home_sharpe_3pt_percent_vs_adv',
                  'game_off_reb_percentage_matchup_differential','spread','home_rest_adv']


frame = pd.DataFrame(X_train,columns=X_subset.columns)

#corr_check = X_standardized_df_subset_clean[features].corr()
#sns.heatmap(corr_check, annot=True, cmap='coolwarm')
#sns.pairplot(X_standardized_df_subset_clean[clean_features])

maxx_custom_features = ['spread','home_sharpe_turnover_percentage_vs_adv','12_game_home_opponent_turnover_percentage',
                        '12_game_away_opponent_turnover_percentage', '12_game_away_turnover_percentage','12_game_home_turnover_percentage',
                        'home_sharpe_off_reb_percentage_vs_adv','12_game_home_3pt_percent',
 '12_game_home_2pt_percent',  '12_game_home_opponent_3pt_percent', '12_game_home_opponent_2pt_percent','12_game_away_3pt_percent',
 '12_game_away_2pt_percent',  '12_game_away_opponent_3pt_percent', '12_game_away_opponent_2pt_percent','home_sharpe_3pt_percent_vs_adv','home_sharpe_2pt_percent_vs_adv',
 '5_game_home_3pt_points_above_ex'  ]

og_model_feats =  ['game_turnover_percentage_matchup_differential',
 'spread',
 'home_sharpe_2pt_percent_vs_adv',
 'home_sharpe_3pt_percent_vs_adv',]

og_model_params = {'colsample_bytree': 1,
 'learning_rate': 0.025,
 'max_depth': 3,
 'n_estimators': 50,
 'reg_lambda': 1.5,
 'subsample': 0.8}

adjusted_og_params = {'colsample_bytree': 0.75,
 'learning_rate': 0.025,
 'max_depth': 5,
 'n_estimators': 500,
 'reg_alpha' : 0.025,
# 'reg_lambda': 10,
 'subsample': 0.8}

good_rf_model_feats = ['home_sharpe_2pt_percent_vs_adv',
 'home_sharpe_3pt_percent_vs_adv',
 'home_sharpe_off_reb_percentage_vs_adv',
 'game_turnover_percentage_matchup_differential',
 'spread']

good_rf_model_params = {'colsample_bytree': 0.5,
 'learning_rate': 0.025,
 'max_depth': 3,
 'n_estimators': 150,
 'reg_lambda': 0.5,
 'subsample': 0.6}


# RANDOM FOREST
rf = RandomForestClassifier(max_depth=7, n_estimators=250)
rf.fit(X_train, y_train)

# Extract feature importance from Random Forest
rf_importance = pd.Series(rf.feature_importances_, index=X_subset.columns).sort_values(ascending=False)
rf_features = rf_importance[rf_importance > 0.0175].index
rf_features = ['home_sharpe_2pt_percent_vs_adv','home_sharpe_3pt_percent_vs_adv','home_sharpe_off_reb_percentage_vs_adv',
                'game_turnover_percentage_matchup_differential','spread']#,'home_sharpe_points_per_possession_vs_adv ']


#LASSO - pretty meaningless results

lasso = Lasso(alpha=0.0005)  # adjust the alpha value
lasso.fit(X_train, y_train)

# Extract feature importance from Lasso coefficients
lasso_importance = pd.Series(lasso.coef_, index=X_subset.columns).sort_values(ascending=False)


#top_4 = [x[0] for x in sorted_feature_selection_dict if sorted_feature_selection_dict.index(x) < 10]
#top_4 = ['game_turnover_percentage_matchup_differential','spread','home_sharpe_2pt_percent_vs_adv','home_sharpe_3pt_percent_vs_adv','home_sharpe_3pt_rate_vs_adv']
#top_4 = ['home_sharpe_3pt_rate_vs_adv','home_sharpe_2pt_rate_vs_adv','home_sharpe_2pt_percent_vs_adv','home_sharpe_points_per_possession_vs_adv','spread]
#top_4 = rf_features
ma_columns.append('home_spread')

X_top_4 = X_subset[ma_columns]




#X_top_4 = X_standardized_df_subset_clean[rf_features]
"""

oppo_frame = pd.DataFrame()
for col in X_standardized_df_subset_clean.columns:
    if col == 'spread':
        oppo_frame[col] = X_top_4[col]
    oppo_frame[col] = -1 * X_top_4[col]

X_top_4 = pd.concat([X_top_4,oppo_frame])
y_oppo = abs(y_subset_clean - 1)

y_subset_clean = y_subset_clean.append(y_oppo, ignore_index=True)"""


#X_train, X_test, y_train, y_test = train_test_split(X_pca_12, y_subset_clean, test_size=0.25, random_state=24)


X_train, X_test, y_train, y_test = train_test_split(X_top_4, y_subset, test_size=0.25, random_state=24)
X_test_unscaled = X_test.copy()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



##############################################
  # MODEL EXECUTION AND GRID SEARCH
##############################################

xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=24)

best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=24, **adjusted_og_params)

#multi_class_model = XGBClassifier(objective='multi:softmax', num_class=len(y_subset.unique()))


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2,3,4],
    'learning_rate': [0.025, 0.05],
    'subsample': [0.8],
    'colsample_bytree': [1.0],
   #'reg_alpha': [0,0.01, 0.05],
   'reg_lambda': [0, 0.5, 0.8, 1, 1.5, 2.5],
    'min_child_weight':[2,4,6]
    #'booster': ['gbtree', 'gblinear', 'dart']
}


total_grid = {
    'n_estimators': [50,100,150],
    'max_depth': [3,5,8],
    #'learning_rate': [0.005, 0.025,0.05],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.5,0.7,1],
    #'reg_lambda': [0.5, 1.5],
    #'min_child_weight': [4,8,12],
    #'reg_alpha': [0.01, 0.1,0.2]
}


grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)

grid_search.fit(X_train, y_train)
best_model.fit(X_train, y_train)

#best_params = grid_search.best_params_
#best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)



# MULTI CLASS MDOEL EVALUATION

"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
"""

##############################################
  # MODEL STRENGTH REVIEW
##############################################

train_pred = best_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print(f'Train Accuracy: {train_acc} \nTest Accuracy: {test_acc}')

probs = best_model.predict_proba(X_test)
probs = probs.tolist()
testVals = y_test.values.tolist()
awayHighRight = []
homeHighRight = []
lowConviction = 0
for i in range(0,len(probs)):
    if probs[i][0] > 0.55:
        if testVals[i] == 0:
            awayHighRight.append(1)
        else:
            awayHighRight.append(0)
    elif probs[i][1] > 0.55:
        if testVals[i] == 1:
            homeHighRight.append(1)
        else:
            homeHighRight.append(0)
    else:
        lowConviction += 1

homeAcc = sum(homeHighRight) / len(homeHighRight)
awayAcc = sum(awayHighRight) / len(awayHighRight)
gameNums = [len(homeHighRight), len(awayHighRight)]
print(gameNums)
print(homeAcc,awayAcc)


# Nonconformal Predictor based on academ paper
"""
from nonconformist.icp import IcpClassifier
from nonconformist.nc import NcFactory

# Create a nonconformity function based on the XGBoost model
nc = NcFactory.create_nc(best_model)  # best_model is your trained XGBoost model

# Create an inductive conformal classifier
icp = IcpClassifier(nc)

# Split your calibration set from the training set
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit the model
icp.fit(X_train, y_train)

# Calibrate the model
icp.calibrate(X_calib, y_calib)

# Make predictions with confidence measures
prediction_intervals = icp.predict(X_test, significance=0.05)

# Convert conformal prediction results to a DataFrame for easier analysis
cp_df = pd.DataFrame(prediction_intervals, index=X_test_unscaled.index)

# Add columns for true and predicted labels
cp_df['true_label'] = y_test
cp_df['predicted_label'] = y_pred

# Coverage: Proportion of true labels within the prediction intervals
coverage = cp_df.apply(lambda row: row['true_label'] in row[row == True].index, axis=1).mean()
print(f"Coverage: {coverage}")

# Correct Predictions within Interval
correct_within_interval = cp_df.apply(lambda row: row['predicted_label'] in row[row == True].index and row['predicted_label'] == row['true_label'], axis=1).mean()
print(f"Correct Predictions within Interval: {correct_within_interval}")

"""


feature_importances = best_model.feature_importances_

feature_names = X_subset[rel_corrs.index].columns.tolist()

feature_importance_dict = dict(zip(feature_names, feature_importances))

### PREDICT PROBA WINNER AND CONVERT TO SPREADS
filename = 'spread_linear_regressor.sav'
spread_pickle = pickle.load(open(filename, 'rb'))
away_probs = [p[0] for p in probs]
home_probs = [p[1] for p in probs]

test_frame = data.loc[y_test.index]
test_frame['away_model_win_prob'] = away_probs
test_frame['home_model_win_prob'] = home_probs
win_eval_frame = test_frame[['spread','o/u','moneyLineHome','moneyLineAway',
                             'home_favor','home_spread_cover','away_model_win_prob','home_model_win_prob']]
model_spreads = spread_pickle.predict(win_eval_frame[['home_model_win_prob','o/u']].values)

win_eval_frame['model_imp_spread'] = model_spreads

sig_diff = 10

win_eval_frame['predict_minus_actual'] = win_eval_frame['model_imp_spread'] - win_eval_frame['spread']
win_eval_frame.loc[win_eval_frame['predict_minus_actual'] >= sig_diff, 'spread_signal'] = 1
win_eval_frame.loc[win_eval_frame['predict_minus_actual'] <= -sig_diff, 'spread_signal'] = -1
win_eval_frame['spread_signal'] = win_eval_frame['spread_signal'].fillna(0)

home_cheap = win_eval_frame[win_eval_frame['spread_signal'] == 1] # bet home
home_rich = win_eval_frame[win_eval_frame['spread_signal'] == -1]  #bet away 56% but a lot of recent data

home_cheap.home_spread_cover.sum() / home_cheap.shape[0]
1- (w.home_spread_cover.sum() / home_rich.shape[0])

results_list = [[home_cheap.home_spread_cover.sum() / home_cheap.shape[0],home_cheap.shape[0] ]
                ,[1- (home_rich.home_spread_cover.sum() / home_rich.shape[0]),home_rich.shape[0] ]]
results_list

#get probs of spreads, add to to X_train, pull data with that index, and compare





#filename = 'model_0919_matchup_diffs.sav'
#pickle.dump(best_model, open(filename, 'wb'))
path = f""
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

joblib.dump(my_scaler, 'scaler.gz')
my_scaler = joblib.load('scaler.gz')




##############################################
  # MODEL PREDICTIONS  ANALYSIS
##############################################

train_result_analysis = pd.DataFrame()
train_result_analysis['pred'] = train_pred
train_result_analysis['result'] = y_train.values.tolist()
train_probs = loaded_model.predict_proba(X_train)
train_probs = [max(i) for i in train_probs]
train_result_analysis['prob'] = train_probs



test_result_analysis = pd.DataFrame()
test_result_analysis['pred'] = test_pred
test_result_analysis['result'] = y_test.values.tolist()
test_probs = loaded_model.predict_proba(X_test)
test_probs = [max(i) for i in test_probs]
test_result_analysis['prob'] = test_probs

train_result_analysis.to_csv('')
test_result_analysis.to_csv('')


X_train['pred'] = train_result_analysis['pred']
X_train['result'] = train_result_analysis['result']
X_train['prob'] = train_result_analysis['prob']

X_test['pred'] = train_result_analysis['pred']
X_train['result'] = test_result_analysis['result']
X_test['prob'] = test_result_analysis['prob']




##############################################
  # CHECKING ANY DATASET
##############################################


data['datetime_date'] = pd.to_datetime(data['finalDate'])
data['month'] = data['datetime_date'].dt.month

old_data = data[(data['regSeason'] == 1) &  (data['season'] == 2023) ] 
                 #(data['Unnamed: 0.2'] >= 11134)]

league_average_columns = [col for col in old_data.columns if 'league_average' in col or 'avg' in col.lower()]
old_data = old_data.drop(columns=league_average_columns)

# Filtering rows where both conditions are true
old_data = old_data[(old_data['12_game_home_3pt_rate'] != 10) &
                                      (old_data['12_game_away_3pt_rate'] != 10) &
                                      (old_data['rolling_home_defense_relative_3pt_rate'] < 1.5) &
                                      (old_data['rolling_away_defense_relative_3pt_rate'] <1.5) &
                                      (old_data['away_total_shot_attempts'] > 60) &
                                      (old_data['home_total_shot_attempts'] > 60) & 
                                      (old_data['spread_difference'] != 0) &
                                      (old_data['spread'] <= 20) &
                                      (old_data['o/u'] <= 275) &
                                      (old_data['o/u'] >= 150)]


matchup_diff_columns = [col for col in old_data.columns if 'matchup_differential' in col]
spread_columns = ['spread','home_spread_cover','o/u']
sharpe_cols = [col for col in old_data.columns if '_adv' in col]

old_matchup = old_data[matchup_diff_columns + sharpe_cols +  spread_columns ]

old_subset = old_matchup.copy()
old_subset = old_subset.replace([np.inf, -np.inf], np.nan).dropna()
old_subset = old_subset.reset_index()


old_x = old_subset.drop(columns=['home_spread_cover','index'])
old_y = old_subset['home_spread_cover']

old_x_data = scaler.transform(old_x)

old_x_final = pd.DataFrame(old_x_data, columns=old_x.columns)

feature_cols = ["game_turnover_percentage_matchup_differential","spread","home_sharpe_2pt_percent_vs_adv","home_sharpe_3pt_percent_vs_adv"]
rf_features = ['home_sharpe_2pt_percent_vs_adv',
 'home_sharpe_3pt_percent_vs_adv',
 'home_sharpe_off_reb_percentage_vs_adv',
 'game_turnover_percentage_matchup_differential',
 'spread']
old_x_final = old_x_final[X_standardized_df_subset_clean.columns]

#loaded_model = pickle.load(open(filename, 'rb'))

old_predict = best_model.predict(old_x_final)
train_acc = accuracy_score(old_y, old_predict)

probs = best_model.predict_proba(old_x_final)
probs = probs.tolist()
testVals = old_y.values.tolist()
awayHighRight = []
homeHighRight = []
lowConviction = 0
for i in range(0,len(probs)):
    if probs[i][0] > 0.55:
        if testVals[i] == 0:
            awayHighRight.append(1)
        else:
            awayHighRight.append(0)
    elif probs[i][1] > 0.55:
        if testVals[i] == 1:
            homeHighRight.append(1)
        else:
            homeHighRight.append(0)
    else:
        lowConviction += 1

homeAcc = sum(homeHighRight) / len(homeHighRight)
awayAcc = sum(awayHighRight) / len(awayHighRight)
gameNums = [len(homeHighRight), len(awayHighRight)]
print(gameNums)
print(homeAcc,awayAcc)



matchup_data_subset = matchup_spread_frame.copy()


X_subset = matchup_data_subset.drop(columns=['home_spread_cover'])
y_subset = matchup_data_subset['home_spread_cover']

####### RAW CLUSTERING - has shown no results so far

# Finding the optimal number of clusters using silhouette score
sil_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_top_4)
    cluster_labels = kmeans.labels_
    sil_score = silhouette_score(X_top_4, cluster_labels)
    sil_scores.append(sil_score)

optimal_clusters = np.argmax(sil_scores) + 1 
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_top_4)
X_top_4['game_cluster'] = clusters
X_top_4 = X_top_4.reset_index()
y_subset_clean = y_subset_clean.reset_index()
X_top_4['result'] = y_subset_clean['home_spread_cover']

xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=9)

for cluster in list(X_top_4.game_cluster.unique()):
    c_frame = X_top_4[X_top_4['game_cluster'] == cluster]
    print(f'{cluster} has {c_frame.result.value_counts()} results')

    c_frame = c_frame.drop(columns=['game_cluster'])
    y_subset = c_frame['result']
    X_subset = c_frame.drop(columns=['result'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=9)

    cluster_grid = {'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5]
    #'reg_lambda': [1, 2]

    }

    grid_search = GridSearchCV(xgb_classifier, cluster_grid, cv=5, scoring='accuracy',n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Getting the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    probs = best_model.predict_proba(X_test)
    probs = probs.tolist()
    testVals = y_test.values.tolist()
    awayHighRight = []
    homeHighRight = []
    lowConviction = 0
    for i in range(0,len(probs)):
        if probs[i][0] > 0.5238:
            if testVals[i] == 0:
                awayHighRight.append(1)
            else:
                awayHighRight.append(0)
        elif probs[i][1] > 0.5238:
            if testVals[i] == 1:
                homeHighRight.append(1)
            else:
                homeHighRight.append(0)
        else:
            lowConviction += 1

    homeAcc = sum(homeHighRight) / len(homeHighRight)
    awayAcc = sum(awayHighRight) / len(awayHighRight)
    gameNums = [len(homeHighRight), len(awayHighRight)]
    print(cluster)
    print(gameNums)
    print(homeAcc,awayAcc)
    print(sum(test_pred))
    print('###########')

feature_importances = best_model.feature_importances_

feature_names = X_train.columns.tolist()

feature_importance_dict = dict(zip(feature_names, feature_importances))




