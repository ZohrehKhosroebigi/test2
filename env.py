from load_data import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Load_show_data import *
from Normalization import *
from create_placeholder import CreatePlaceholder
from init_params import InitParams
from tensorflow.python.framework import ops
def init_params(params, w, dim_w):
    # with tf.variable_scope(w,reuse = tf.AUTO_REUSE):
    init_w = tf.get_variable(w, dim_w, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    params[w] = init_w
    return params
# test Laod_show_data
#test norm
print("L3====================")
L3 = Loaddata()
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = L3.load
L3.showpic(3)
print(L3)
print("L4====================")
L4=NoramlPic(L3.load)
L4.norm()
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
#test
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