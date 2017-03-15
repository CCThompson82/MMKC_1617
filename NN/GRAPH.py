"""This is the graph for the March Madness Kaggle Competition"""



madness = tf.Graph()

with madness.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Dense_layers') :
            with tf.name_scope('fc_1') :
                W_1 = tf.Variable(tf.truncated_normal([num_features, fc_depth[0]], stddev = stddev ))
                b_1 = tf.Variable(tf.zeros([fc_depth[0]]))
                tf.summary.histogram('W_1', W_1)
                tf.summary.histogram('b_1', b_1)
            with tf.name_scope('fc_2') :
                W_2 = tf.Variable(tf.truncated_normal([fc_depth[0], fc_depth[1]], stddev = stddev ))
                b_2 = tf.Variable(tf.zeros([fc_depth[1]]))
                tf.summary.histogram('W_2', W_2)
                tf.summary.histogram('b_2', b_2)
            with tf.name_scope('fc_3') :
                W_3 = tf.Variable(tf.truncated_normal([fc_depth[1], fc_depth[2]], stddev = stddev ))
                b_3 = tf.Variable(tf.zeros([fc_depth[2]]))
                tf.summary.histogram('W_3', W_3)
                tf.summary.histogram('b_3', b_3)

        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[2], num_labels], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('W_clf', W_clf)
            tf.summary.histogram('b_clf', b_clf)


    def dense_layers(data, keep_prob) :
        """
        Executes a series of dense layers.
        """
        def fc(data, W, b) :
            """Convenience function for relu dense layer with dropout"""
            fc = tf.nn.dropout(
                    tf.nn.relu(
                        tf.matmul(data, W) + b),
                    keep_prob)
            return fc

        d1 = fc(data, W_1, b_1)
        d2 = fc(d1, W_2, b_2)
        d3 = fc(d2, W_3, b_3)
        return d3




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            X = tf.placeholder(tf.float32, shape = [batch_size, num_features])
            score_labels = tf.placeholder(tf.float32, shape = [batch_size])
            win_labels = tf.placeholder(tf.float32, shape = [batch_size,num_labels])
            learning_rate = tf.placeholder(tf.float32, shape = () )
            beta = tf.placeholder(tf.float32, shape = () )

        with tf.name_scope('Network') :
            dense_output = dense_layers(X, keep_prob = keep_prob)
        with tf.name_scope('Classifier') :
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation') :
            regularization = (beta*tf.nn.l2_loss(W_1) +
                             beta*tf.nn.l2_loss(W_2) +
                             beta*tf.nn.l2_loss(W_3) +
                             beta*tf.nn.l2_loss(W_clf))

            xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits = logits, labels = win_labels)
            cross_entropy = tf.reduce_mean(xent)
            cost = cross_entropy #+ regularization

            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('Validation') :
        with tf.name_scope('Input') :
            tourney_X = tf.constant(valid_X, dtype = tf.float32)
            tourney_y = tf.constant(valid_y, dtype = tf.float32)
            tourney_scores = tf.constant(valid_scores, dtype = tf.float32)
        with tf.name_scope('Network') :
            train_dense_output = dense_layers(X, keep_prob = 1.0)
            valid_dense_output = dense_layers(tourney_X, keep_prob = 1.0)
        with tf.name_scope('Prediction') :
            train_logits = tf.nn.softmax(tf.matmul(train_dense_output, W_clf) + b_clf)
            valid_logits = tf.nn.softmax(tf.matmul(valid_dense_output, W_clf) + b_clf)
        with tf.name_scope('Assessment') :
            train_xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits = train_logits, labels = win_labels))
            valid_xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits = valid_logits, labels = tourney_y))


    with tf.name_scope('Summaries') :
        tf.summary.scalar('Regularization', regularization)
        tf.summary.scalar('Cross_entropy_during_GD', cross_entropy)
        tf.summary.scalar('Cost', cost)
        tf.summary.scalar('Regularization_beta', beta)
        tf.summary.scalar('Learning_rate', learning_rate)
        tf.summary.scalar('Cross_entropy_Train', train_xent)
        tf.summary.scalar('Cross_entropy_Valid', valid_xent)
        summaries = tf.summary.merge_all()
