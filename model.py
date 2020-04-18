from load_data import load_dataset
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Load_show_data import *
from Normalization import *
from create_placeholder import CreatePlaceholder
from init_params import InitParams
from tensorflow.python.framework import ops
from forward_propag import ForwardPorpagation
from compute_cost import Costfunc
from mini_bach import random_mini_batches
class Model():
    def model(self,X_train, Y_train, X_test, Y_test, classes, learning_rate,
              num_epochs, minibatch_size, print_cost):
        """
        Implements a three-layer ConvNet in Tensorflow:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        Arguments:
        X_train -- training set, of shape (None, 64, 64, 3)
        Y_train -- test set, of shape (None, n_y = 6)
        X_test -- training set, of shape (None, 64, 64, 3)
        Y_test -- test set, of shape (None, n_y = 6)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        test_accuracy -- real number, testing accuracy on the test set (X_test)
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
        seed = 3  # to keep results consistent (numpy seed)
        (m, n_H0, n_W0, n_C0) = X_train.shape
        n_y = Y_train.shape[1]
        costs = []  # To keep track of the cost
        #create obj placeholder
        placeholder=CreatePlaceholder()
        # Create Placeholders of the correct shape
        ### START CODE HERE ### (1 line)
        placeholder.createplaceholder(n_H0, n_W0, n_C0, n_y)
        ### END CODE HERE ###

        # Initialize parameters
        initparams=InitParams()
        initparams.init_params("w_1", "w1", [4, 4, 3, 8])
        initparams.init_params("w_2", "w2", [2, 2, 8, 16])
        #parameters = initialize_parameters()
        ### END CODE HERE ###

        # Forward propagation: Build the forward propagation in the tensorflow graph
        forward = ForwardPorpagation()
        # since it is an input in each conv
        X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad = placeholder.X, initparams.params["w1"], [1, 1, 1, 1], 'SAME', [1, 8, 8, 1], [1, 8, 8, 1], 'SAME'
        forward.forward(X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad)

        # second layer
        new_X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad = forward.layer["P1"], initparams.params["w2"], [1, 1, 1,1], 'SAME', [1, 4, 4, 1], [1, 4, 4, 1], 'SAME'
        forward.forward(new_X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad)

        # full_connected(self,P,classes):
        forward.full_connected(len(classes))
        #Z3 = forward_propagation(X, parameters)
        print("ZZZZ", forward.layer["Z1"].shape)
        ### END CODE HERE ###

        # Cost function: Add cost function to tensorflow graph
        ### START CODE HERE ### (1 line)
        cst = Costfunc()
        cst.cost(forward.layer["Z1"], placeholder.Y)
        #cost = compute_cost(Z3, Y)
        ### END CODE HERE ###

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        ### START CODE HERE ### (1 line)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cst.cost_)
        ### END CODE HERE ###

        # Initialize all the variables globally
        init = tf.global_variables_initializer()
        logfile = open('logs/log.txt', 'a')
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess_mdl:

            # Run the initialization
            sess_mdl.run(init)

            # Do the training loop
            for epoch in range(num_epochs):

                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    """
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost.
                    # The feedict should contain a minibatch for (X,Y).
                    """
                    ### START CODE HERE ### (1 line)
                    _, temp_cost = sess_mdl.run([optimizer, cst.cost_], feed_dict={placeholder.X: minibatch_X, placeholder.Y: minibatch_Y})
                    ### END CODE HERE ###

                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                    logfile.write("<-----" + str(datetime.datetime.now()) + "----->" + ' \n')
                    logfile.write("Cost after epoch:    " +str((epoch, minibatch_cost))+'\n')

                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            plt.savefig('images/test.png')
            # Calculate the correct predictions
            predict_op = tf.argmax(forward.layer["Z1"], 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(placeholder.Y, 1))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            logfile.write("accuracy:    " + str(accuracy) + '\n')
            print(accuracy)
            train_accuracy = accuracy.eval({placeholder.X: X_train, placeholder.Y: Y_train})
            test_accuracy = accuracy.eval({placeholder.X: X_test, placeholder.Y: Y_test})
            logfile.write("Train Accuracy:    " + str(train_accuracy) + '\n')
            logfile.write("Test Accuracy:    " + str(test_accuracy) + '\n')
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
            logfile.write("parameters or weights:    " + str(initparams.params) + '\n')
            return train_accuracy, test_accuracy, initparams.params
