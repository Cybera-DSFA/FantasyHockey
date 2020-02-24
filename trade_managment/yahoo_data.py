from yahoo_oauth import OAuth2
import json
import yahoo_fantasy_api as yfa


def leag(idx = -1, file = '../trade_managment/oauth2.json'):
    '''
    function to just return the leauge to abstract 
    away the authentication in other files
    '''
    oauth = OAuth2(None, None, from_file=file)
    g = yfa.Game(oauth, 'nhl')
    lg = yfa.League(oauth, g.league_ids()[-1])
    return lg

def get_team_keys_and_names(leauge):

    '''
    function to get the leauge team and keys
    '''
    team_keys = leauge.teams()
    tk = []
    team_names = []
    for key in team_keys:
        tk.append(key)
        team_names.append(team_keys[key]['name'])
    return tk, team_names
    

def get_team_players(leauge, team_key):
    '''
    Function to return players on the team, if they are undroppable
    and their current status
    '''
    roster = leauge.to_team(team_key).roster()
    players = []
    undroppable = []
    status = []
    for dic in roster:
        try:
            player = leauge.player_details(dic['name'])[0]
        except:
            print(dic)
            continue
        players.append(player['name']['full'])
        if player['is_undroppable'] == '1':
            undroppable.append(player['name']['full'])

        try:
            if (player['status'] != '' ):
                if player['status'] == 'DTD':
                    continue
                else:
                    status.append(player['name']['full'])
        except:
            continue
    return players, undroppable, status


def get_taken_players(leauge):
    '''
    function to return a list of players that are taken
    '''
    tp = []
    taken_players = leauge.taken_players()
    for dic in taken_players:
        tp.append(dic['name'])
    return tp

def get_waiver_players(leauge):
    '''
    Function to return a list of players on waivers
    '''
    waiver_players = leauge.waivers()
    wp = []
    for dic in waiver_players:
        wp.append(dic['name'])
    return wp

def get_out_free_agents(leauge):
    '''
    function to return a list of players that are free agents, 
    but also injured so we don'twant them
    '''
    out = []
    positions = ['C', 'G', 'D', 'LW', 'RW']
    for pos in positions:
        print("Finding", pos)
        players = leauge.free_agents(pos)
        for dic in players:
            if dic['status'] != '':
                if dic['status'] != "DTD":
                    out.append(dic['name'])
    return out


    
