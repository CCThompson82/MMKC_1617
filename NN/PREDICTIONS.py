"""This is the session call associated with GRAPH.py"""

wd = os.getcwd()
md = wd+'/MODELS/'+version_ID
if not os.path.exists(md) :
    os.makedirs(md)
tensorboard_path = md+'/Tensorboard_logs'



with tf.Session(graph = madness) as session :


    print("Loading version {}".format(version_ID))
    with open(md+'/meta_dictionary.pickle', 'rb') as  handle :
        meta_dict = pickle.load(handle)
    print("Metadata dictionary loaded!")
    epochs_completed = meta_dict.get(np.max([key for key in meta_dict])).get('Num_epochs')
    total_games = meta_dict.get(np.max([key for key in meta_dict])).get('games_trained')

    restorer = tf.train.Saver()
    print("Initializing restorer...")
    restorer.restore(session, tf.train.latest_checkpoint(md))
    print("Weights and biases retrieved!  Picking up at {} epochs completed : {} training images observed".format(epochs_completed, total_games))


    print("\nPredicting matches from the NCAA 2016-17 Season Tournament...")

    print("\nTRAINING MM Predictor...")

    preds = session.run(pred_logits)
