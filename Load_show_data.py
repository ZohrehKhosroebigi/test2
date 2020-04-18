from load_data import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class Loaddata():##  Load data
    import load_data
    X_train_orig=""
    Y_train_orig=""
    X_test_orig=""
    Y_test_orig=""
    classes=""
    @property
    def load(self):
        Loaddata.X_train_orig,Loaddata.Y_train_orig,Loaddata.X_test_orig,Loaddata.Y_test_orig,Loaddata.classes =load_dataset()
        return Loaddata.X_train_orig,Loaddata.Y_train_orig,Loaddata.X_test_orig,Loaddata.Y_test_orig,Loaddata.classes
#show an image of x_train to user

    def showpic(self,idx):
        print(Loaddata.X_train_orig[idx].shape)
        plt.imshow(Loaddata.X_train_orig[idx])
        plt.show()
        print(f'y= {np.squeeze(Loaddata.Y_train_orig[:,idx])}')


    def __str__(self):
        try:
            return f'Dataset inclueds: X_train_orig shape {Loaddata.X_train_orig.shape}\n,Y_train_orig shape{Loaddata.Y_train_orig.shape}\n,X_test_orig shape{Loaddata.X_test_orig.shape}\n,Y_test_orig shape {Loaddata.Y_test_orig.shape}\n, Classes is: {Loaddata.classes}\n, number of training examples= {Loaddata.X_train_orig.shape[0]}\n,number of test examples= {Loaddata.X_test_orig.shape[0]}'
        except Exception as err:
            print(err)

    def __repr__(self):
        try:
            return f'{Loaddata.X_train_orig},{Loaddata.Y_train_orig},{Loaddata.X_test_orig},{Loaddata.Y_test_orig}, {Loaddata.classes}, {Loaddata.X_train_orig.shape[0]},{Loaddata.X_test_orig.shape[0]}'
        except Exception as err:
            print(err)