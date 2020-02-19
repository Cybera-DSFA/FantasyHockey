import pandas as pd
import  json
from urllib.request import urlopen
import sys
sys.path.insert(1, './')
import scripts.hockey_bots as hockey 

def stats_list(pid):
    '''
    This is a bit of cheater function to get the fields of stats that are tracked. 
    Needs a player id for a goalie or a non-goalie to get both fields'''
    response = urlopen('https://statsapi.web.nhl.com/api/v1/people/' +str(pid) + '/stats?stats=gameLog&season=20192020')
    d = json.load(response)
    
    stat = []
    for dic in d['stats'][0]['splits']:
        for key in dic['stat']:
            stat.append(key)
        break
    return stat

def gather_teams(): 
    '''
    This gathers all the team information as well as the players taht are on each team
    '''
    team_url = 'https://statsapi.web.nhl.com/api/v1/teams/'

    response = urlopen(team_url)
    d = json.load(response)

    team_dict = {'team_name':[], 'team_id':[]}
    for team in d['teams']:
        team_dict['team_name'].append(team['name'])
        team_dict['team_id'].append(team['id'])
    x = pd.DataFrame(team_dict)
    team_dict2 = {'team_name':[], 'team_id':[], 'fullName':[], 'player_id':[], 'position':[]}
    
    for id_ in team_dict['team_id']:
        response = urlopen(team_url + str(id_) + '/roster')
        d = json.load(response)
        
        for ob in d['roster']:
            team_dict2['team_name'].append(x[x.team_id == id_].team_name.values[0])
            team_dict2['team_id'].append(id_)
            team_dict2['player_id'].append(ob['person']['id'])
            team_dict2['fullName'].append(ob['person']['fullName'])
            team_dict2['position'].append(ob['position']['abbreviation'])
    
    return pd.DataFrame(team_dict2) 


def stat_gather(player_df, pos, stat, season = '20192020'):

    '''
    This Gathers all the stats for each player for each game in a specific season
    and returns a data frame
    '''
    player_scores = {}
    for keys in stat:
        player_scores[keys] = []

    player_scores['player_id'] = []
    player_scores['fullName'] = []
    player_scores['game_id'] = []
    player_scores['team_name'] = []
    player_scores['team_id'] = []
    
    if pos == 'G':
        print("Gathering Goalie Stats")
        ids = player_df[player_df.position == 'G'].player_id.values
    else:
        print("Gathering Non Goalie Stats")
        ids = player_df[player_df.position != 'G'].player_id.values
    
    for pid in ids:
        response = urlopen('https://statsapi.web.nhl.com/api/v1/people/'+str(pid)+'/stats?stats=gameLog&season='+season)
        d = json.load(response)
        for dic in d['stats'][0]['splits']:
            player_scores['game_id'].extend([dic['game']['gamePk']])
            player_scores['player_id'].extend([pid])
            player_scores['fullName'].extend([player_df[player_df.player_id == pid].fullName.values[0]])
            player_scores['team_id'].extend([dic['team']['id']])
            player_scores['team_name'].extend([dic['team']['name']])
            for key in stat:
                try:
                    player_scores[key].extend([dic['stat'][key]])
                except KeyError:
                    player_scores[key].extend([None])

    return pd.DataFrame(player_scores)

def combine_frames(team_df, player_df, goalie_df):
    # add game number to players
    play_df = player_df.merge(team_df[['fullName', 'position']], on = 'fullName')
    
    d2 = play_df.groupby(['game_id','team_name']).first().reset_index()
    d2['game_num'] = d2.groupby('team_name').cumcount() + 1
    play_df = pd.merge(d2[['game_id', 'game_num', 'team_name']], play_df, on = ['game_id', 'team_name'])
    # add game number to goalies 
    d2 = goalie_df.groupby(['game_id','team_name']).first().reset_index()
    d2['game_num'] = d2.groupby('team_name').cumcount() + 1
    goalie_df = pd.merge(d2[['game_id', 'game_num', 'team_name']], goalie_df, on = ['game_id', 'team_name'])
    
    x = pd.concat([play_df, goalie_df], ignore_index=True, axis=0)
    return x

def game_fill(df):
    games = df.game_num.unique().tolist()
    players = df.fullName.unique().tolist()
    fill_dict = {}
    
    safe_key = ['game_id', 'game_num', 'position', 'team_name', 'fullName', 'player_id']
    for key in list(df):
        fill_dict[key] = []
        
    for player in players:
        games_played = df[df.fullName == player].game_num.tolist()
        fill_game = list(set(games) - set(games_played))
        position = df[df.fullName==player].position.tolist()[0]
        pid = df[df.fullName==player].player_id.tolist()[0]
        team = df[df.fullName==player].team_name.tolist()[0]
        for game in fill_game:
            # print(len(fill_game))
            try:
                fill_dict['game_id'].extend([df[(df.team_name == team) & (df.game_num == game)].game_id.tolist()[0]])
            except Exception as e:
                # print(e)
                fill_dict['game_id'].extend(['game_has_not_happened'])
            fill_dict['game_num'].extend([game]) 
            fill_dict['position'].extend([position])
            fill_dict['team_name'].extend([team])
            fill_dict['fullName'].extend([player])
            fill_dict['player_id'].extend([pid])
            for key in fill_dict:
                if key not in safe_key:
                    fill_dict[key].extend([None])
            
            
            
    return pd.DataFrame(fill_dict)

def score(row):
    if row.position == "G":
        return hockey.goalie_points(row, row.gamesStarted)
    else:
        return hockey.player_points(row)

def get_data(season = '20192020', return_separate=False):
    print("downloading team data")
    team_df = gather_teams()
    # update this so we don't have to worry about retirement 
    g_stats = stats_list(8475839)
    p_stats = stats_list(8477949)
    goalie_df = stat_gather(team_df, pos='G', stat=g_stats)
    player_df = stat_gather(team_df, pos='ddfsdf', stat=p_stats, season = season )
    
    fin_frame = combine_frames(team_df, player_df, goalie_df) 
    
    print("filling in missing games")
    df = fin_frame.copy()
    append_df = game_fill(df)
    
    f = fin_frame.append(append_df, ignore_index=True).fillna(0)
    print("calculating fantasy points")
    f['score'] = f.apply(score, axis=1)

    f = f.dropna()
    if return_separate:
        return team_df, g_stats, p_stats, goalie_df, player_df, fin_frame, f

    return f

    