import numpy as np 
import pandas as pd 
import cvxpy as cp 
import sys 
import json
import requests
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

def generateConferenceLists():
    with open('../data/processed/teams.json') as f:
        d = json.load(f)
    team = pd.Series(d).reset_index()
    team.columns = ['teamid', 'teamname']

    east = ['Boston Bruins', 'Tampa Bay Lightning', 'Washington Capitals',
                    'Philadelphia Flyers', 'Pittsburgh Penguins', 'Carolina Hurricanes',
                    'New York Islanders','Toronto Maple Leafs','Columbus Blue Jackets',
                    'Florida Panthers', 'New York Rangers','Montreal Canadiens']

    west=['St. Louis Blues', 'Colorado Avalanche','Vegas Golden Knights',
                    'Dallas Stars', 'Edmonton Oilers', 'Nashville Predators',
                    'Vancouver Canucks','Calgary Flames', 'Winnipeg Jets', 
                    'Minnesota Wild', 'Arizona Coyotes', 'Chicago Blackhawks']
    t = team[team.teamname.isin(east + west)]


    east_list = list(t[t.teamname.isin(east)].teamid.unique()) 
    west_list = t[t.teamname.isin(west)].teamid.unique()

    return east_list, west_list

def conference(row, east_list, west_list):
    if str(row.team)  in east_list:
        return "E"
    if str(row.team)  in west_list:
        return "W"

def conferenceIndex(div, df):
    df_chose = df.groupby('player_id').first().reset_index()
    df_chose = df_chose[df_chose['div'].str.contains(div)]
    return df_chose.index.tolist()    

def sportnet_optim(scores, 
                      taken, 
                      mine, 
                      gammaa, 
                      df,
                      edefence,
                      wdefence,
                      egoalie,
                      wgoalie,
                      eforward,
                      wforward,
                      team_size = 12,
                      ed = 2,
                      wd=2,
                      eg = 1,
                      wg =1,
                      ef = 3,
                      wf = 3, max_salary=30):
    '''
    This function solves the binary linear programming problem 
    max(r^T x - gamma x^T Q x) where r is the average score per game for a player,
    X is a binary player vector, gamma is the "risk tolerance" parameter, and Q is the 
    covariance matrix of all the scores for each palyer. This is also subject 
    to certain constraints such as maximum salary, player numbers, and number of players
    in a given postion.
    '''
  
    x = cp.Variable(len(scores.mean()),boolean = True)
    
    gamma = cp.Parameter(nonneg=True)
    ret = np.array(scores.mean()).T * x
   
    sigma = np.array(scores.cov())
    risk = cp.quad_form(x, sigma)
    
    constraints = []
    # Here we are putitng in the max value constraint which give us a cost
    # associated with each player
    S = np.diag(df.groupby('player_id').max().PV.tolist())
    # L1 norm here, absolute value is fine as no salaries should be negative.
    constraints.append(cp.sum(S @ x) <= max_salary )
    # Cannot pick taken players
    for i in range(len(taken)):
        if taken[i] not in mine:
            constraints.append(x[taken[i]] == 0)
    # Must pick players we already have chosen 
    for i in range(len(mine)):
        constraints.append(x[mine[i]] == 1)

    constraints = constraints + [cp.sum(x) == team_size,
               cp.sum(x[edefence]) == ed,
               cp.sum(x[wdefence]) == wd,
               cp.sum(x[egoalie]) == eg,
               cp.sum(x[wgoalie]) == wg,
               cp.sum(x[eforward]) == ef,
               cp.sum(x[wforward]) == wf
               ] 


    # actually defining our problem 
    prob = cp.Problem(cp.Maximize(ret - gamma*risk),
                   constraints)
   
    gamma.value = gammaa
    # TODO: we can probably tighten some of these up 
    # Note: after this x is defined as our players, 
    prob.solve(parallel=True,   
               mi_max_iters=500,
               mi_abs_eps = 1e-4,
               mi_rel_eps = 1e-1,
               max_iters=200,
               abstol = 1e-5,
               reltol = 1e-4,
               feastol = 1e-5,
               abstol_inacc = 5e-3,
               reltol_inacc = 5e-3,
               feastol_inacc = 1e-2)
    print(prob.status)
    players = list(np.where(x.value.round(1) ==1)[0])
    risk_data = cp.sqrt(risk).value
    return_data = ret.value
    return players, risk_data, return_data

def displayTeam(df, team, all_points):
    return df[df.player_id.isin(list(all_points.iloc[:,team]))].drop_duplicates(subset='name')

def ram_selection(players, scores, df, g): 
    ohboy = df[df.player_id.isin(list(scores.iloc[:,players]))]
    playerids = ohboy.groupby('player_id').count().index
    pid = ohboy.groupby('player_id').count()
    R = np.array(scores[playerids].mean())
    Q = np.array(scores[playerids].cov())
    x = cp.Variable(len(players))
    
    gamma = cp.Parameter(nonneg=True)
    ret = R.T * x 
    risk = cp.quad_form(x, Q)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
                   [cp.sum(x) == 1, 
                    x >= 0])
    
    gamma.value = g
    prob.solve()
    indexes = list(range(len(players)))
    # Find the index of the player who has the largest proportion of 
    # investment, provided they haven't already been selected 
    # then find the next largest 
    print(sorted(x.value)[::-1])
    return playerids[x.value.argsort()[::-1]]#players[np.array(playerids)[list(x.value.argsort()[::-1])]]

def name_extract(row):
    place = row.index
    if "Forwards" in place:
        kind = 'Forwards'
    if "Defence" in place:
        kind = 'Defence'
    if "Goalies" in place:
        kind = "Goalies"
    name = re.findall('[A-Z][a-z]*', row[kind])
    return name[1] + ' ' + name[0]

def valueScraper(username, password, url = 'https://fantasy.sportsnet.ca/sportsnet/hkplayoff20/enter_picks'):
    '''
    This function is a scraper to pull the points values of each player
    off of sportsnet. You'll need to provide your own username and password
    to the function to sign in to yoursports net account if you're going to use this. 
    '''
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    time.sleep(5)
    username = driver.find_element_by_id("capture_signIn_traditionalSignIn_emailAddress")
    password = driver.find_element_by_id("capture_signIn_traditionalSignIn_password")

    username.send_keys(username)
    password.send_keys(password)
    time.sleep(3)
    driver.find_element_by_id("capture_signIn_traditionalSignIn_signInButton").click()
    time.sleep(3)

    html = driver.page_source

    soup = BeautifulSoup(html,'html.parser')
    res = soup.findAll("div")     
    driver.close()
    x = pd.read_html(html)

    for i in range(1, 7):
        x[i]['name'] = x[i].apply(name_extract, axis=1)
        values = values.append(x[i][['name', 'PV']], ignore_index=True)

    return values 