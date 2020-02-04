import pandas as pd 
import numpy as np
import cvxpy as cp
from heapq import nlargest

def position_indexes(all_pos, all_points, df, idx, position):
    '''
    finding indexes in aggregate data of each player
    in a particular position
    '''
    homes = []
    for idx_ in all_points[player_constraint(position, df, idx)]:
        homes.append(all_pos.index(idx_))
    return homes

def player_merge(df_players, df_games, df_player_info, df_salaries):
    '''
    This function is used to merge the dataframe of players, the games the've played,
    some meta data and their salaries all into one, more impressive, dataframe
    '''
    df_ = pd.merge(df_players, df_games[['game_id', 'date_time', 'type']])
    df_['date_time'] = pd.to_datetime(df_['date_time'])
    
    df_ = pd.merge(df_, df_player_info[['player_id','firstName', 'lastName', 'primaryPosition']])
    # df_ = pd.merge(df_, df_salaries[['firstName', 'lastName', 'Salary']], on = ['firstName', 'lastName'])
    
    return df_

def player_points(row, 
           goal = 5, 
           assit_divisor = 1.7,
           pm = 0.66,
           shot = 0.45, 
           block = 0.24, 
           face = 0.07,
           penaltymult = -0.32,
           short_hand = 0.59):
    '''
    This function implements a custom player point scoring scheme. 
    '''
    row = row.copy()
    ass_mult = goal/assit_divisor
    goals = goal * row.goals
    ass_points = ass_mult * row.assists
    plus_minus = pm * row.plusMinus
    shot_score = shot * row.shots
    blocks = block * row.blocked
    # counts face off wins and losses (losses = total - wins)
    faceOffs = face * ( 2 * row.faceOffWins - row.faceoffTaken)
    penalty = penaltymult * row.penaltyMinutes
    shortHanded = row.shortHandedGoals * short_hand * goal
    shortHanded = shortHanded + row.shortHandedAssists * goal/assit_divisor * short_hand
    
    return goals + ass_points + plus_minus + shot_score + blocks + faceOffs + shortHanded + penalty

def goalie_points(row,
                  goal_shifts,
                  save = 0.45,
                  assist = 5/1.7,
                  goal_against = -5,
                  start_score = 8.66):

    '''
    This function is to implement the custom scoring system for goalies. Goal_shifts is a 
    dataframe to determine if a goalie started a game or not . 
    '''
    
    row = row.copy()
    save_points = row.saves * save
    ass_points = row.assists * assist
    goals_in = (row.shots - row.saves) * goal_against
    started = goal_shifts[(goal_shifts.game_id == row.game_id) &
                   (goal_shifts.player_id == row.player_id)]['shift_start'].values
    if 0 in started:
        start = start_score
    else:
        start = 0
    
    return save_points + ass_points + goals_in + start


def player_constraint(position, df, idx ):
    '''
    this function actually finds all players given a certain postition and their indexes
    (for scoring) as idx. This _technically_ forms a constraint, but not directly like
    the name here may imply. Is this a bad naming convention? Yes. Should I change it? Also yes.
    Will I? Probably not. 
    '''
    df_chose = df.groupby('player_id').max()
    df_chose = df_chose[df_chose.primaryPosition.str.contains(position)]
    return df_chose.index.tolist()

def salary_constraint(x, df, idx):
    '''
    this function actually finds all players salaries given a certain postition and their indexes
    (for scoring) as idx. This is only used to help construct a diagonal matrix if salary constraints
    are implemented in the league 
    '''
    chosen = np.nonzero(x)[0]
    chosen = np.array(idx)[chosen]
    df_chose = df[df.player_id.isin(chosen)]
    money = df_chose.groupby('player_id').max().Salary.sum()
    return money

def optimize_choice(players, scores, df, g, taken, mine): 
    ohboy = df[df.player_id.isin(list(scores.iloc[:,players]))]
    playerids = ohboy.groupby('player_id').count().index
    
    test = scores.mean()
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
    largest = nlargest(len(players), indexes, key=lambda i: x.value[i])
    for i in range(len(largest)):
        pot = players[largest[i]]
       # print(pot, taken)
        if pot in taken:
            continue
        if pot in mine:
            continue
        else: 
            mine.append(pot)
            taken.append(pot)
            return mine, taken, pot

def optim_player(scores, 
                      taken, 
                      mine, 
                      gammaa, 
                      df,
                      defence,
                      center,
                      goalie,
                      right_wingers,
                      left_wingers,
                      sub_gamma = .5, 
                      selection = "max",
                      max_salary = False,
                      team_size = 17,
                      min_d = 4,
                      min_g = 2,
                      min_c = 2,
                      min_rw = 2,
                      min_lw =2,
                      full_team = False):
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
    # Cannot pick taken players
    for i in range(len(taken)):
        if taken[i] not in mine:
            constraints.append(x[taken[i]] == 0)
    # Must pick players we already have chosen 
    for i in range(len(mine)):
        constraints.append(x[mine[i]] == 1)
    # Add the salary constraint if we need to 
    if max_salary:
        S = np.diag(df.groupby('player_id').max().Salary.tolist())/10000000
         # L1 norm here, absolute value is fine as no salaries should be negative.
        constraints.append(cp.norm(S @ x, p=1) <= max_salary)
    
    constraints = constraints + [cp.sum(x) == team_size,
                   cp.sum(x[defence]) >=min_d,
                   cp.sum(x[goalie]) >= min_g,
                   cp.sum(x[goalie]) <= min_g + 1,
                   cp.sum(x[center]) >= min_c,
                   cp.sum(x[right_wingers]) >= min_rw,
                   cp.sum(x[left_wingers]) >= min_lw] 
                
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
    
    # Pick highest score player
    # TODO: Update picking stradegy to also include as option sqrt(scores.mean**2 + scores.std**2)
    # Also add _another_ optimization problem with just the team and pick the one with the highest
    # investment proportion (no longer binary - but just with the players we have chosen)
    
    # finding which indexes are non zero (within floating point)
    players = list(np.where(x.value.round(1) ==1)[0])
    new_players = [x for x in players if x not in mine]
    if full_team:
        risk_data = cp.sqrt(risk).value
        return_data = ret.value
        return players, risk_data, return_data

    if selection == 'max':
       
        possible = np.take(np.array(scores.mean()), new_players)
        to_take = list(scores.mean()).index(max(possible))

        mine.append(to_take)
        taken.append(to_take)
         # print('max', to_take)
        return mine, taken, to_take
    
    if selection == 'rms':
        mean_ = scores.mean()
        std_ = scores.std()
        values_ = np.sqrt(mean_**2 + std_**2)
        possible = np.take(values_, new_players)
        to_take = list(values_).index(max(possible))
        #print(to_take)
       # print(to_take, "rms")
        mine.append(to_take)
        taken.append(to_take)
        
        return mine, taken, to_take

    if selection == 'optim':
         mine, taken, to_take = optimize_choice(players, scores, df, sub_gamma, taken, mine)
         #print(to_take, 'optim')
         return mine, taken, to_take

def greedy_competitor(all_points, 
                      taken, 
                      mine, 
                      defence,
                      center,
                      goalie,
                      right_wingers,
                      left_wingers):
    '''
    a function that simulates a player with the strategy of "always pick the player available
    with the highest available returns, provided I have already found the minimum number of players
    required in each position"
    '''
    remaining_players = list(set(range(len(all_points.mean()))) - set(taken))
    scores = np.take(np.array(all_points.mean()), remaining_players)
    sorted_scores = sorted(scores, reverse=True)

    for i in range(len(scores)):
        pot_max = sorted_scores[i]
        pot_player = list(scores).index(sorted_scores[i])
        if pot_player in taken:
            continue
        if pot_player in defence:
            trigger = 'defence'
            if len(mine['defence']) < 4 :
                mine['defence'].append(pot_player)
                taken.append(pot_player)
                return mine, taken, pot_player
       
        if pot_player in center:
            trigger = 'center'
            if len(mine['center']) < 2:
                mine['center'].append(pot_player)
                taken.append(pot_player)
                return mine, taken, pot_player
            
        if pot_player in goalie:
            trigger = 'goalie'
            if len(mine['goalie']) < 2:
                mine['goalie'].append(pot_player)
                taken.append(pot_player)
                return mine, taken, pot_player
            
        if pot_player in right_wingers:
            trigger = 'right_winger'
            if len(mine['right_winger']) < 2:
                mine['right_winger'].append(pot_player)
                taken.append(pot_player)
                return mine, taken, pot_player
            
        if pot_player in left_wingers:
            trigger = 'left_winger'
            if len(mine['left_winger']) < 2:
                mine['left_winger'].append(pot_player)
                taken.append(pot_player)
                return mine, taken, pot_player
        else:
            # TODO: We need to fix this so it gets a full roster
            mine[trigger].append(pot_player)
            taken.append(pot_player)
            return mine, taken, pot_player
def input_name():
    while True:
        name = input("Please enter player name ")
        print(len(name.strip().split()))
        if len(name.strip().split()) == 2:
            print('chosing', name)
            return name
        else: 
            print("incorrect name format, try again")
            continue
            
            
def human(df_, all_points, name, taken, mine):
    '''
    a function for manual entry and seletion of players 
    if competing against people
    '''
    # in case there's a new player not in the optimization 
    
    while True:
        first, last = name.strip().split(" ")
        df = df_[(df_.firstName.str.contains(first, case=False)) & 
                   (df_.lastName.str.contains(last, case=False))]
        
        if len(df.game_id) == 0:
            print("empty data frame?", name)
            if name == "ROOKIE OVERRIDE":
                return mine, taken
            else:
                print("empty data frame? spelling mistake most likely")
                name = input_name()
            continue
        else: 
            p = df['player_id'].unique()[0]
            df2 = all_points.mean().reset_index()
            player_index = list(df2[df2['player_id'] == p].index)[0]
            if player_index in taken:
                print('player alread taken, try another')
                name = input_name()
                continue
            mine.append(player_index)
            taken.append(player_index)
            break
    print(mine, taken)
    return mine, taken

def draft(functions, order, team_size=17, pause = False, team_names = None,  **kwargs):
    '''  
    This function is to run a draft which decides on a team. The 'functions' argument
    is a list of functions (defined above) which can be used to simulate players, and 
    order is the order in which those functions (players) will draft. Note that 
    this order is automatically reversed during the draft process. 
    '''

    greedy_selections = kwargs['greedy_selections']
    taken = []
    # the teams
    mine = [[] for i in range(len(functions))]
    df = kwargs['df']
    all_points = kwargs['scores']
    for i in range(team_size):
        print("Beginning round", i)
        for j in order:
            #print(j)
            if team_names:
                print(team_names[j])
            if functions[j].__name__ == 'optim_player':
                mine[j], taken, to_take = functions[j](scores=kwargs['scores'],
                                                       df=kwargs['df'],
                                                       taken=taken, 
                                                       mine=mine[j],
                                                       gammaa=kwargs['gammaa'][j],
                                                       defence=kwargs['defence'],
                                                       center=kwargs['center'],
                                                       goalie=kwargs['goalie'],
                                                       selection=kwargs['selection'][j],
                                                       right_wingers=kwargs['right_wingers'],
                                                       left_wingers=kwargs['left_wingers'],
                                                       sub_gamma=kwargs['sub_gamma'][j])
                print("Optim Player order ", j, " with")
                print("Gamma = ", kwargs['gammaa'][j], "selection = ", kwargs['selection'][j])
                
                ohboy = df[df.player_id.isin(list(all_points.iloc[:,[to_take]]))]
                playerids = ohboy.groupby('player_id').count().index
                n = df[df.player_id.isin(playerids)][['firstName', 'lastName', 'primaryPosition']].drop_duplicates().values
                print("Chose player: ", n[0][0], n[0][1], n[0][2])
                print()
                if pause:
                    input("Press enter to continue")

            if functions[j].__name__ == 'greedy_competitor':
                # only one greedy boi allowed atm
                #print('greedy')
                index_of_greed = j
                greedy_selections, taken, chosen = functions[j](all_points=kwargs['scores'], 
                                                                taken=taken,
                                                                mine=greedy_selections,
                                                                defence=kwargs['defence'],
                                                                center=kwargs['center'],
                                                                goalie=kwargs['goalie'],
                                                                right_wingers=kwargs['right_wingers'],
                                                                left_wingers=kwargs['left_wingers'])
                print("Greedy Player order ", j)
                print(chosen)
                ohboy = df[df.player_id.isin(list(all_points.iloc[:,[chosen]]))]
                playerids = ohboy.groupby('player_id').count().index
                n = df[df.player_id.isin(playerids)][['firstName', 'lastName', 'primaryPosition']].drop_duplicates().values
                print("Chose player: ", n[0][0], n[0][1],n[0][2])
                print()
            if functions[j].__name__ == 'human':
                name = input_name()
                mine[j], taken = human(df_ = kwargs['df'],
                                       all_points = kwargs['scores'], 
                                       name=name, 
                                       taken=taken, 
                                       mine=mine[j])


            
                
        order = order[::-1]
    # gotta unwrap the dictionary 
    # if greedy_selections: 
    # for key in greedy_selections:
    #     mine[index_of_greed] += greedy_selections[key]
    
    return taken, mine