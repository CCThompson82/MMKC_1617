"""This is the session call associated with GRAPH.py"""

wd = os.getcwd()
md = wd+'/MODELS/'+version_ID
if not os.path.exists(md) :
    os.makedirs(md)
tensorboard_path = md+'/Tensorboard_logs'



with tf.Session(graph = madness) as session :

    # check for metadata dictionary
    if 'meta_dictionary.pickle' in os.listdir(md) and initiate_model != True:
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

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")

    else :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        meta_dict = {0 : { 'Num_epochs' : 0,
                            'version_ID' : version_ID,
                            'games_trained' : 0}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")
        epochs_completed = 0
        total_games = 0
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    print("\nTRAINING MM Predictor...")
    while open('NN/stop.txt', 'r').read().strip() != 'True' :
        training_set_list = RgRes.index.tolist()

        while len(training_set_list) >= batch_size :
            batch_X, batch_y, _ = prepare_batch(RgRes, training_set_list, batch_size)

            feed_dict = {   X : batch_X,
                            win_labels : batch_y,
                            learning_rate : float(open('NN/learning_rate.txt', 'r').read().strip()),
                            beta : float(open('NN/beta.txt', 'r').read().strip()),
                        }

            if (total_games % (batch_size*summary_rate)) == 0 :
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)
                writer.add_summary(summary_fetch, total_games)
            else :
                _ = session.run(train_op, feed_dict = feed_dict)

            total_games += batch_size


        epochs_completed += 1
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
        print("Epoch {} completed : {} games observed. Model checkpoint created!".format(epochs_completed, total_games))
        meta_dict[epochs_completed] = { 'Num_epochs' : epochs_completed,
                            'version_ID' : version_ID,
                            'games_trained' : total_games}

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)
