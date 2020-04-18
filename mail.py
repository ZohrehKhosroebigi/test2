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
#train and test model
np.random.seed(1)
mymodel=Model()
data_orgi=Loaddata()
data_orgi.load
data_orgi.showpic(4)
print("__________orginal data are loaded_________")
data_norm=NoramlPic(data_orgi.load)
data_norm.norm()
mymodel.model(data_norm.X_train,data_norm.Y_train,data_norm.X_test,data_norm.Y_test,data_norm.classes,learning_rate=0.009,num_epochs=1, minibatch_size=64, print_cost=True)
