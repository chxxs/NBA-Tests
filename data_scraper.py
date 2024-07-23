import pandas as pd
#from basketball_reference_scraper.box_scores import get_box_scores
from bs4 import BeautifulSoup
import requests
import time
from datetime import datetime, timedelta



### FOR UPDATING OVER LONGER PERIODS

start_date = datetime(2023, 4, 15)
end_date = datetime(2023, 6, 12)

dates = []

current_date = start_date
while current_date <= end_date:
    dates.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)




#### SPREAD WORK


spread_columns = ['date','awayTeam','homeTeam','opening_spread','spread','finalAway','finalHome']
spread_vals = []


for date in dates:
    spread_url = f'https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={date}'
    response = requests.get(spread_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    parent_div = soup.find('div', id="tbody-nba")
    if parent_div == None:
        continue
    game_divs = parent_div.find_all('div', class_=lambda value: value and 'GameRows_eventMarketGridContainer' in value)
    
    

    for game in game_divs:
        team_divs = game.find_all('span', class_='GameRows_participantBox__0WCRz')

        # Assuming the first one is the away team and the second is the home team
        away_team = team_divs[0].text if team_divs else None
        home_team = team_divs[1].text if len(team_divs) > 1 else None

        opening_spread_element = game.find('span', class_='GameRows_adjust__NZn2m')
        opening_spread = opening_spread_element.text if opening_spread_element else None

        # Find the element with the current spread value
        spread_element = game.find('span', class_='OddsCells_adjust__hGhKV')
        spread = spread_element.text if spread_element else None

        # Find the element with the final away team score
        final_away_element = game.find_all('div', class_='GameRows_scores__YkN24')[0]
        finalAway = final_away_element.text if final_away_element else None

        # Find the element with the final home team score
        final_home_element = game.find_all('div', class_='GameRows_scores__YkN24')[1]
        finalHome = final_home_element.text if final_home_element else None

        spread_vals.append([date,away_team,home_team,opening_spread,
                                spread, finalAway,finalHome])

spread_frame = pd.DataFrame(spread_vals, columns=spread_columns)

#### MONEYLINE WORK

moneyline_vals = []
moneyline_columns = ['away_ml_open','home_ml_open','away_ml','home_ml']

for date in dates:

    moneyline_url = f'https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/full-game/?date={date}'
    response = requests.get(moneyline_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    parent_div = soup.find('div', id="tbody-nba")
    if parent_div == None:
        continue

    game_divs = parent_div.find_all('div', class_=lambda value: value and 'GameRows_eventMarketGridContainer' in value)

    teams = soup.find_all("span", class_="GameRows_participantBox__0WCRz")
    extracted_teams = [team.get_text() for team in teams]

    openers = soup.find_all("span", class_="GameRows_opener__NivKJ")
    extracted_first_odds = [odd.get_text() for odd in openers]
    extracted_first_odds = [val for val in extracted_first_odds if len(val) > 0]
    
    second_odds = soup.select(".OddsCells_oddsNumber__u3rsp.OddsCells_compact__cawia span:nth-of-type(2)")
    extracted_second_odds = [odd.get_text() for odd in second_odds]

    num_games = int(len(teams) / 2)

    for i in range(0, num_games):
        away_team = extracted_teams[i*2]
        home_team = extracted_teams[(i*2) + 1]

        away_opener = extracted_first_odds[i*2]
        home_opener = extracted_first_odds[(i*2)+1]

        away_close = extracted_second_odds[(14*i)] #14
        home_close = extracted_second_odds[((1))+(14*i)] #15

        moneyline_vals.append([away_opener,
                                home_opener, away_close, home_close])
moneyline_frame = pd.DataFrame(moneyline_vals,columns= moneyline_columns)


#### GAME TOTAL WORK

game_total_vals = []
game_total_cols = ['opening_o/u','o/u']


for date in dates:

    total_url = f'https://www.sportsbookreview.com/betting-odds/nba-basketball/totals/full-game/?date={date}'
    response = requests.get(total_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    parent_div = soup.find('div', id="tbody-nba")
    if parent_div == None:
        continue

    game_divs = parent_div.find_all('div', class_=lambda value: value and 'GameRows_eventMarketGridContainer' in value)

    teams = soup.find_all("span", class_="GameRows_participantBox__0WCRz")
    extracted_teams = [team.get_text() for team in teams]

    openers = soup.find_all("span", class_="GameRows_adjust__NZn2m")
    extracted_first_odds = [odd.get_text() for odd in openers]
    extracted_first_odds = [val for val in extracted_first_odds if len(val) > 0]
    
    closing_odds = soup.find_all("span", class_="OddsCells_adjust__hGhKV")
    extracted_second_odds = [odd.get_text() for odd in closing_odds]

    num_games = int(len(teams) / 2)

    for i in range(0, num_games):
        opening_total = extracted_first_odds[i*2]

        closing_total = extracted_second_odds[(14*i)] #14
        game_total_vals.append([opening_total,closing_total])
total_frame = pd.DataFrame(game_total_vals,columns= game_total_cols)


final_frame_cols = ['date', 'awayTeam', 'homeTeam','spread', 'moneyLineHome',
       'moneyLineAway', 'o/u', 'finalAway', 'finalHome', 'homeWin',
       'homeFavor', 'awayFavor', 'pointDiff', 'favoriteWin', 'favoriteLine',
       'equalLine', 'favoriteDollarsWon', 'dogLine', 'dogDollarsWon', 'year',
       'yearDateRight']
all_scraped_odds = pd.concat([spread_frame, moneyline_frame, total_frame], axis=1)
all_scraped_odds.to_csv(f'/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/odds_23_24/{date}_odds.csv')

###

lines = pd.read_csv(f'/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/odds_23_24/{prev_date}_odds.csv')

all_lines = pd.concat([lines,all_scraped_odds])
all_lines.to_csv(f'/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/odds_23_24/{date}_odds.csv')



### TAKES IN RECENT LINES CSV HERE AND SLICES TO TODAY'S DATE

### SOME PREPROCESSING

teams = all_lines['awayTeam'].values.tolist()
teams = list(set(teams))
teams.sort()
abbreviations = ['ATL','BOS','BRK','CHO','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL',
                 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

teamMatcher = dict(zip(teams, abbreviations))

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','DRB','TOV','PTS']
col1 = stats
col1 = ['away_' + i for i in col1]

col2 = stats
col2 = ['home_' + i for i in col2]

lineCols = lines.columns.tolist()
allCols = lineCols + col1 + col2
masterFrame = pd.DataFrame(columns = allCols)
all_master = pd.read_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/odds_23_24/games_{}.csv'.format(prev_date),index= True,header=True)



rowsNotProcessed = []
noMatchGames = 0
rowsProcessed = 0
#datatype = new.dtypes['finalDate']
brokenGames = []


# FULL GAME BUILD
for index, row in lines.iterrows():
    bbref_date = row.date.replace('-', '')
    needToBreak = 0
    rowVals = row.tolist()
    if row['finalAway'] == 0:
        allVals = rowVals + ([1] * 20)
        masterFrame.loc[len(masterFrame)] = allVals
        continue
    away = teamMatcher[row['awayTeam']]
    home = teamMatcher[row['homeTeam']]
    link = f'https://www.basketball-reference.com/boxscores/{bbref_date}0{home}.html'
    response = requests.get(link)
    linkNotWork = 0
    while response.status_code != 200:
        print('{} vs {} on {} NOT WORKING'.format(away, home, row['finalDate']))
        time.sleep(30)
        response = requests.get(link)
        linkNotWork += 1
        if linkNotWork >= 2:
            noMatchGames += 1
            print('{} vs {} on {} DID NOT PROCESS'.format(away, home, row['finalDate']))
            rowsNotProcessed.append(rowsProcessed)
            rowsProcessed += 1
            needToBreak = 1
            break
    if needToBreak == 1:
        continue
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    table = soup.find('table')

    try:
        awayTable = soup.find('table', id="box-{}-game-basic".format(away))
        awayFrame = pd.read_html(str(awayTable))[0]
    except ValueError:
        brokenGames.append([row['finalDate'], row['awayTeam']])
        continue

    awayFrame = awayFrame.droplevel(level=0, axis=1)

    homeTable = soup.find('table', id="box-{}-game-basic".format(home))
    homeFrame = pd.read_html(str(homeTable))[0]
    homeFrame = homeFrame.droplevel(level=0, axis=1)

    awayCols = awayFrame.columns.values.tolist()
    homeCols = homeFrame.columns.values.tolist()

    awayTeamTotals = awayFrame.loc[awayFrame['Starters'] == 'Team Totals']
    homeTeamTotals = homeFrame.loc[homeFrame['Starters'] == 'Team Totals']
    totals = [awayTeamTotals, homeTeamTotals]
    teamStats = []
    for team in totals:
        for stat in stats:
            teamStats.append(team[stat].iloc[0])
    allVals = rowVals + teamStats # + [1,'2023']

    masterFrame.loc[len(masterFrame)] = allVals
    print('Game {}: {} vs {} done on {}'.format(rowsProcessed, row['awayTeam'], row['homeTeam'], row['date']))
    rowsProcessed += 1
    if (rowsProcessed % 12) == 0:
        time.sleep(30)
    if noMatchGames >= 10:
        break

final_master = pd.concat([all_master,masterFrame])


final_master.to_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/odds_23_24/games_{}.csv'.format(date),index= True,header=True)


### FIRST HALF BUILD 

for index, row in lines.iterrows():
    if row['finalDate'] == '2016-12-31':
        print('debug')
    needToBreak = 0
    rowVals = row.tolist()
    away = teamMatcher[row['awayTeam']]
    home = teamMatcher[row['homeTeam']]
    link = 'https://www.basketball-reference.com/boxscores/{}0{}.html'.format(row['yearDateRight'],home)
    response = requests.get(link)
    linkNotWork = 0
    while response.status_code != 200:
        print('{} vs {} on {} NOT WORKING'.format(away,home,row['finalDate']))
        time.sleep(30)
        response = requests.get(link)
        linkNotWork += 1
        if linkNotWork >= 2:
            noMatchGames += 1
            print('{} vs {} on {} DID NOT PROCESS'.format(away, home, row['finalDate']))
            rowsNotProcessed.append(rowsProcessed)
            rowsProcessed += 1
            needToBreak = 1
            break
    if needToBreak == 1:
        continue
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    table = soup.find('table')

    try:
        awayTable = soup.find('table', id="box-{}-h1-basic".format(away))
        awayFrame = pd.read_html(str(awayTable))[0]
    except ValueError:
        brokenGames.append([row['finalDate'],row['awayTeam']])
        continue
     #change this to 0 for full game, 3 for first half
    awayFrame = awayFrame.droplevel(level=0, axis=1)

    homeTable = soup.find('table',id="box-{}-h1-basic".format(home))
    homeFrame = pd.read_html(str(homeTable))[0]
    homeFrame = homeFrame.droplevel(level=0, axis=1)
    """statType = list(homeFrame.columns.values.tolist()[1])[0]
    homeFrame = homeFrame.droplevel(level=0, axis=1)

    awayPlayer = awayFrame['Starters'].iloc[0]
    homePlayer = homeFrame['Starters'].iloc[0]
    counter = 1
    while homePlayer == awayPlayer:
        resultIndex = 8 + counter
        homeFrame = pd.read_html(str(allTables))[resultIndex]
        homeFrame = homeFrame.droplevel(level=0, axis=1)
        homePlayer = homeFrame['Starters'].iloc[0]
        statType = list(homeFrame.columns.values.tolist()[1])[0]"""
    awayCols = awayFrame.columns.values.tolist()
    homeCols = homeFrame.columns.values.tolist()
    """frameIndex = 6
    while 'FG' not in homeCols:
        homeFrame = pd.read_html(str(allTables))[frameIndex]
        homeFrame = homeFrame.droplevel(level=0, axis=1)
        homeCols = homeFrame.columns.values.tolist()
        frameIndex += 1
        if frameIndex > 16:
            break"""
    awayTeamTotals = awayFrame.loc[awayFrame['Starters'] == 'Team Totals']
    homeTeamTotals = homeFrame.loc[homeFrame['Starters'] == 'Team Totals']
    totals = [awayTeamTotals, homeTeamTotals]
    teamStats = []
    for team in totals:
        for stat in stats:
            teamStats.append(team[stat].iloc[0])
    allVals = rowVals + teamStats
    masterFrame.loc[len(masterFrame)] = allVals
    print('Game {}: {} vs {} done on {}'.format(rowsProcessed, row['awayTeam'],row['homeTeam'],row['finalDate']))
    rowsProcessed += 1
    if (rowsProcessed % 12) == 0:
        time.sleep(30)
    if noMatchGames >= 10:
        break
    """if index % 1000 == 0:
        masterFrame.to_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/firstHalfFirst{}.csv'.format(index),
                           index=True, header=True)"""



masterFrame.to_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/firstHalfStatsTo_12_31_16.csv',index= True,header=True)
print(masterFrame.head())

"""    try:
        boxDict = get_box_scores(row['finalDate'], away, home, period='GAME', stat_type='BASIC')
    except ValueError:
        print('{} vs {} DID NOT WORK on {}'.format(row['awayTeam'],row['homeTeam'],row['finalDate']))
        noMatchGames.append(row)
        continue
    except AttributeError:
        print('{} vs {} DID NOT WORK on {}'.format(row['awayTeam'],row['homeTeam'],row['finalDate']))
        noMatchGames.append(row)
        continue
    if not bool(boxDict):
        noMatchGames.append(row)
        print('{} vs {} DID NOT WORK on {}'.format(row['awayTeam'], row['homeTeam'], row['finalDate']))
        continue
    try:
        aFrame = boxDict[away]
        hFrame = boxDict[home]
    except TypeError:
        noMatchGames.append(row)
        print('{} vs {} DID NOT WORK on {}'.format(row['awayTeam'], row['homeTeam'], row['finalDate']))
        continue

    #todo make two lists of away and home and add those together to make frame
    aFrame = aFrame.loc[aFrame['PLAYER'] == 'Team Totals']
    hFrame = aFrame.loc[aFrame['PLAYER'] == 'Team Totals']

    frames = [aFrame, hFrame]
    boxVals = []
    for frame in frames:
        for stat in stats:
            statVal = aFrame[stat].iloc[0]
            boxVals.append(statVal)
    allVals = rowVals + boxVals"""




### OVER TIME GAME FIX
#link = 'https://www.basketball-reference.com/boxscores/201412080WAS.html'
"""link = 'https://www.basketball-reference.com/boxscores/201701010ATL.html'
response = requests.get(link)
text = response.text
soup = BeautifulSoup(text, 'html.parser')
table = soup.find('table')
table = soup.find('table', id="box-WAS-game-basic")
awayTable = soup.find('table', id="box-{}-h1-basic".format('SAS'))
frame = pd.read_html(str(awayTable))[0]
allTables = soup.find_all('table')
awayFrame = pd.read_html(str(table))[0]
awayFrame = awayFrame.droplevel(level=0,axis=1)

awayPlayer = awayFrame['Starters'].iloc[0]
homeFrame = pd.read_html(str(allTables))[8]
level0 = list(homeFrame.columns.values.tolist()[1])[0]
for x in range(0,24):
    testFrame = pd.read_html(str(allTables))[x]
    testFrame.to_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/homeFixV{}.csv'.format(x), index=True, header=True)

awayFrame = awayFrame.droplevel(level=0,axis=1)
homeFrame = homeFrame.droplevel(level=0,axis=1)
homeFrame.to_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/homeFixV1.csv',index= True,header=True)
"""
"""link = 'https://www.basketball-reference.com/boxscores/201410280SAS.html'
response = requests.get(link)
yy = response.text
soup = BeautifulSoup(yy,'html.parser')
table = soup.find('table')
allTables = soup.find_all('table')
raw_df = pd.read_html(str(table))[0]
spurs = pd.read_html(str(allTables))[8]"""




###OLD FULL GAME BUILD
"""


for index, row in lines.iterrows():
    


    needToBreak = 0
    rowVals = row.tolist()
    away = teamMatcher[row['awayTeam']]
    home = teamMatcher[row['homeTeam']]
    link = 'https://www.basketball-reference.com/boxscores/{}0{}.html'.format(row['yearDateRight'],home)
    response = requests.get(link)
    linkNotWork = 0
    while response.status_code != 200:
        print('{} vs {} on {} NOT WORKING'.format(away,home,row['finalDate']))
        time.sleep(30)
        response = requests.get(link)
        linkNotWork += 1
        if linkNotWork >= 2:
            noMatchGames += 1
            print('{} vs {} on {} DID NOT PROCESS'.format(away, home, row['finalDate']))
            rowsNotProcessed.append(rowsProcessed)
            rowsProcessed += 1
            needToBreak = 1
            break
    if needToBreak == 1:
        continue
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    table = soup.find('table')
    allTables = soup.find_all('table')

    awayFrame = pd.read_html(str(table))[0]  #todo change this to 0 for full game, 3 for first half
    awayFrame = awayFrame.droplevel(level=0, axis=1)
    #todo if len of tags greater than 16, home needs to be 9 not 8, check if alltables greater than 16, add if ot column
    #todo above was incorrect fix to get all the overtime games parsing correctly, best to look for player
    homeFrame = pd.read_html(str(allTables))[8]
    statType = list(homeFrame.columns.values.tolist()[1])[0]
    homeFrame = homeFrame.droplevel(level=0, axis=1)

    awayPlayer = awayFrame['Starters'].iloc[0]
    homePlayer = homeFrame['Starters'].iloc[0]
    counter = 1
    while homePlayer == awayPlayer:
        resultIndex = 8 + counter
        homeFrame = pd.read_html(str(allTables))[resultIndex]
        homeFrame = homeFrame.droplevel(level=0, axis=1)
        homePlayer = homeFrame['Starters'].iloc[0]
        statType = list(homeFrame.columns.values.tolist()[1])[0]




    #homeMins = homeFrame.loc[homeFrame['Starters'] == 'Team Totals']['MP'].values.tolist()[0]

    awayCols = awayFrame.columns.values.tolist()
    homeCols = homeFrame.columns.values.tolist()
    frameIndex = 6
    while 'FG' not in homeCols:
        homeFrame = pd.read_html(str(allTables))[frameIndex]
        homeFrame = homeFrame.droplevel(level=0, axis=1)
        homeCols = homeFrame.columns.values.tolist()
        frameIndex += 1
        if frameIndex > 16:
            break
    awayTeamTotals = awayFrame.loc[awayFrame['Starters'] == 'Team Totals']
    homeTeamTotals = homeFrame.loc[homeFrame['Starters'] == 'Team Totals']
    totals = [awayTeamTotals, homeTeamTotals]
    teamStats = []
    for team in totals:
        for stat in stats:
            teamStats.append(team[stat].iloc[0])
    allVals = rowVals + teamStats
    masterFrame.loc[len(masterFrame)] = allVals
    print('Game {}: {} vs {} done on {}'.format(rowsProcessed, row['awayTeam'],row['homeTeam'],row['finalDate']))
    rowsProcessed += 1
    if (rowsProcessed % 12) == 0:
        time.sleep(30)
    if noMatchGames >= 10:
        break"""



###old date and name fixing

"""lines = lines.replace('LA Clippers','LAClippers')
lines = lines.replace('LACLippers','LAClippers')
lines = lines.replace('Oklahoma City','OklahomaCity')
lines = lines.replace('Golden State','GoldenState')

today = '0410'
yesterday = '0305'
yearDateYest = '2023' + yesterday
yearDateToday = int('2023'+today)
yearDateYest = int(yearDateYest)
#lines = pd.read_csv('011623odds.csv')
#lines = pd.read_csv('2023_lines_to_{}.csv'.format(today))
lines = pd.read_csv('/Users/maxxestrin/Desktop/Models/NBA_Historical_Odds/gpt_builds/manual_2023_lines.csv')

yestRow =  lines[lines['yearDateRight'] == yearDateYest].index
todayRow = lines[lines['yearDateRight'] == yearDateToday].index


newLines = lines.iloc[list(yestRow)[0]:,]


"""

"""for x in range(100):
    b = get_box_scores('2022-11-05', 'HOU', 'MIN', period='GAME', stat_type='ADVANCED')
    if not bool(b):
        b = get_box_scores('2022-11-05', 'HOU', 'MIN', period='GAME', stat_type='ADVANCED')
        print('Not work')
    else:
        print('worked')"""
