def parse_result(df, game_index) :
    """
    Converts a given result into a 1-dimensional array composed of the
    home (A) and away (B) team season statistics, retrieved from the stats dict.
    Function requires the `id2team_dict` and `stats_dict` to be available in the global environment.
    """
    col_order = ['Wins','Losses','Pyth','AdjustO', 'AdjustD','AdjustT',
                 'SOS Pyth','NCSOS Pyth', 'SOS OppO','SOS OppD','Luck']

    global stats_dict
    global id2team_dict

    # Find the entry in the results df
    game = df.ix[game_index, :]

    # define home team (A) (if match is neutral then first team is 'A')
    if game['Wloc'] == 'H' :
        home_id = game['Wteam']
        away_id = game['Lteam']
        neutral = np.array([-0.5])
        home_net = game['Wscore'] - game['Lscore']
    elif game['Wloc'] == 'A' :
        home_id = game['Lteam']
        away_id = game['Wteam']
        neutral = np.array([-0.5])
        home_net = game['Lscore'] - game['Wscore']
    elif game['Wloc'] == 'N' :
        neutral = np.array([0.5])
        home_id = game['Wteam']
        away_id = game['Lteam']
        home_net = game['Wscore'] - game['Lscore']
    else :
        return("ERROR in result retrieval")

    # retrieve stats in order of home and away in same order of categories
    home = id2team_dict.get(home_id)
    away = id2team_dict.get(away_id)
    ## retrieve dictionaries of stats

    home_stats_d = stats_dict.get(game['Season']).get(home)
    away_stats_d = stats_dict.get(game['Season']).get(away)

    ## convert to ordered array
    home_stats = np.array(pd.Series(home_stats_d).ix[col_order])
    away_stats = np.array(pd.Series(home_stats_d).ix[col_order])

    # combine arrays into single example for training set
    ret_vec = np.concatenate([neutral, home_stats, away_stats], axis = 0)

    if np.isnan(ret_vec).any() == True :
        print('Error in Season {}, with {} or {}'.format(game['Season'], home, away))
        print("see index: {} for more info".format(game_index))
        print(ret_vec)

    # determine result
    y = int(home_net > 0)
    return np.expand_dims(ret_vec, 0), np.expand_dims(home_net, 0), np.expand_dims(y, 0)

def prepare_batch(df, epoch_ix_list, batch_size) :
    """
    Utilizes parse_result fn to return arrays of data and labels.
    """
    assert len(epoch_ix_list) >= batch_size, "Not enough examples left in this epoch to fill batch_size"

    global stats_dict
    global id2team_dict

    for i in range(batch_size) :
        ix = epoch_ix_list.pop(np.random.randint(0, len(epoch_ix_list)))
        match_arr, y_score, y = parse_result(df, ix)
        #while np.isnan(match_arr).any() == True : #Hack to find all the key issues
        #    ix = epoch_ix_list.pop(np.random.randint(0, len(epoch_ix_list)))
        #    match_arr, y_score, y = parse_result(df, ix)
        if y[0] == 1 : # hack to make easy onehot encoding
            y = np.array([[1, 0]])
        else :
            y = np.array([[0, 1]])

        if i == 0 :
            batch_arr = match_arr
            batch_y = y
            batch_score = y_score
        else :
            batch_arr = np.concatenate([batch_arr, match_arr], 0)
            batch_y = np.concatenate([batch_y, y], 0)
            batch_score = np.concatenate([batch_score, y_score], 0)
        del match_arr, y_score, y
    return batch_arr, batch_y, batch_score
