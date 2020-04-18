from load_data import load_dataset
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
from model import Model
from mini_bach import random_mini_batches

"""
def init_params(params, w, dim_w):
    # with tf.variable_scope(w,reuse = tf.AUTO_REUSE):
    init_w = tf.get_variable(w, dim_w, initializer=tf.contrib.layers.xavier_initializer())
    params[w] = init_w
    return params
"""
# test Laod_show_data
#test norm

"""
print("L3====================")
L3 = Loaddata()
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = L3.load
L3.showpic(3)
print(L3)
print("L4====================")
L4=NoramlPic(L3.load)
X_train_1, Y_train_1, X_test_1, Y_test_1, classes=L4.norm()
print(L4)
#test placeholder
print("L5-----------------------")
L5=CreatePlaceholder()
L5.createplaceholder(64,64,3,6)
print(L5)
#test initparams
print("L6-----------------------")
#initialization of parameters:
my = InitParams()
# init_params(self,w_1,"w1",[4, 4, 3, 8]):
my.init_params("w_1", "w1", [4, 4, 3, 8])
my.init_params("w_2", "w2", [2, 2, 8, 16])
#test init
#disable temprory
"""
"""
print("Strat TF_________________")
tf.reset_default_graph()
with tf.Session().as_default() as sess_test:
    my = InitParams()
    #init_params(self,w_1,"w1",[4, 4, 3, 8]):
    my.init_params("w_1","w1",[4, 4, 3, 8])
    my.init_params("w_2", "w2", [2,2,8,16])
    print("+++++++++++++++++"+ str(my.params))
    myinit = tf.global_variables_initializer()
    print("init________________"+str(myinit))
    sess_test.run(myinit)
    #print("WWWWWWWWWWW"+ str(my))
    print("w1111111111111111" + str(my.params["w1"]))
    print("W1[1,1,1] = \n" + str(my.params["w1"].eval()[1,1,1]))
    print("W1.shape: " + str(my.params["w1"].shape))
    print("\n")
    print("w2222222222222222222" + str(my.params["w2"]))
    print("W2[1,1,1] = \n" + str(my.params["w2"].eval()[1,1,1]))
    print("W2.shape: " + str(my.params["w2"].shape))
"""
#Forward test
"""
print("Strat TF Forward_________________")
L3 = Loaddata()
L3.load
#L3.showpic(3)
#print(L3)
#print("L4====================")
L4=NoramlPic(L3.load)
L4.norm()
print(L4)
tf.reset_default_graph()
with tf.Session() as sess_test1:
    L5 = CreatePlaceholder()
    L5.createplaceholder(64, 64, 3, 6)
    my = InitParams()
    # init_params(self,w_1,"w1",[4, 4, 3, 8]):
    my.init_params("w_1", "w1", [4, 4, 3, 8])
    my.init_params("w_2", "w2", [2, 2, 8, 16])
    print("+++++++++++++++++" + str(my.params))
    Fw=ForwardPorpagation()
    #since it is an input in each conv
    X,W,conv_strides,conv_pad,pool_size,pool_strides,pool_pad=L5.X,my.params["w1"],[1,1,1,1],'SAME',[1,8,8,1],[1,8,8,1],'SAME'
    Fw.forward(X,W,conv_strides,conv_pad,pool_size,pool_strides,pool_pad)

    #second layer
    new_X,W,conv_strides,conv_pad,pool_size,pool_strides,pool_pad=Fw.layer["P1"],my.params["w2"],[1,1,1,1],'SAME',[1,4,4,1],[1,4,4,1],'SAME'
    Fw.forward(new_X,W,conv_strides,conv_pad,pool_size,pool_strides,pool_pad)

    # full_connected(self,P,classes):
    Fw.full_connected(len(L4.classes))
    myinit = tf.global_variables_initializer()
    print("init________________" + str(myinit))
    sess_test1.run(myinit)
    # print("WWWWWWWWWWW"+ str(my))
    a=sess_test1.run(Fw.layer["Z1"],{L5.X:np.random.randn(2,64,64,3),L5.Y:np.random.randn(2,6)})
    print("Z3= \n"+str(a))
"""
"""
#test cost
print("Strat TF cost_________________")
tf.reset_default_graph()
with tf.Session() as sess_test1:
    L5 = CreatePlaceholder()
    L5.createplaceholder(64, 64, 3, 6)
    my = InitParams()
    # init_params(self,w_1,"w1",[4, 4, 3, 8]):
    my.init_params("w_1", "w1", [4, 4, 3, 8])
    my.init_params("w_2", "w2", [2, 2, 8, 16])
    print("+++++++++++++++++" + str(my.params))
    Fw = ForwardPorpagation()
    # since it is an input in each conv
    X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad = L5.X, my.params["w1"], [1, 1, 1, 1], 'SAME', [1,8,8,1], [1, 8, 8, 1], 'SAME'
    Fw.forward(X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad)
    # second layer
    new_X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad = Fw.layer["P1"], my.params["w2"], [1, 1, 1,1], 'SAME', [1, 4, 4, 1], [1, 4, 4, 1], 'SAME'
    Fw.forward(new_X, W, conv_strides, conv_pad, pool_size, pool_strides, pool_pad)

    # full_connected(self,P,classes):
    Fw.full_connected(len(classes))
    cst=Costfunc()
    cst.cost(Fw.layer["Z1"],L5.Y)
    myinit = tf.global_variables_initializer()
    print("init________________" + str(myinit))
    sess_test1.run(myinit)
    # print("WWWWWWWWWWW"+ str(my))
    a = sess_test1.run(cst.c, {X: np.random.randn(4, 64, 64, 3), L5.Y: np.random.randn(4, 6)})
    print("Z3= \n" + str(a))
"""
#test model
#batch size
"""
mymodel=Model()
data_orgi=Loaddata()
data_orgi.load
data_norm=NoramlPic(data_orgi.load)
data_norm.norm()
(m, n_H0, n_W0, n_C0) = data_norm.X_train.shape
tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
seed = 3
for epoch in range(1):
    minibatch_cost = 0.
    print("MMMM---" + str(m))

    num_minibatches = int(m / 64)  # number of minibatches of size minibatch_size in the train set
    seed = seed + 1
    print("MMMM---" + str(num_minibatches))
    #minibatches = random_mini_batches(data_norm.X_train,data_norm.Y_train, 64, seed)
"""
mymodel=Model()
data_orgi=Loaddata()
data_orgi.load
#data_orgi.showpic(4)
print("__________orginal data are loaded_________")
data_norm=NoramlPic(data_orgi.load)
data_norm.norm()
mymodel.model(data_norm.X_train,data_norm.Y_train,data_norm.X_test,data_norm.Y_test,data_norm.classes,learning_rate=0.009,num_epochs=100, minibatch_size=64, print_cost=True)
