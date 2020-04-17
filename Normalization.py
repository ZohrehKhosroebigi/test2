from  one_hot import convert_to_one_hot
import tensorflow as tf
class NoramlPic():

    X_train=[]
    Y_train=""
    X_test=""
    Y_test=""
    classes =""
    def __init__(self, myobj):
        try:
            #self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes=myobj
            self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes =  myobj
            #print(self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes)
        except Exception as err:
            print(err)
    def norm(self):

        NoramlPic.X_train=self.X_train/255.
        NoramlPic.X_train = tf.convert_to_tensor(NoramlPic.X_train, dtype=tf.int32)
        NoramlPic.X_test=self.X_test/255.
        NoramlPic.X_test = tf.convert_to_tensor(NoramlPic.X_test, dtype=tf.int32)
        NoramlPic.Y_train=convert_to_one_hot(self.Y_train,len(self.classes)).T
        NoramlPic.Y_test = convert_to_one_hot(self.Y_test, len(self.classes)).T
        NoramlPic.classes=self.classes
        return NoramlPic.X_train,NoramlPic.X_test,NoramlPic.Y_train,NoramlPic.Y_test,NoramlPic.classes

    def __str__(self):
        return f'Dataset inclueds: X_train shape {self.X_train}\nY_train shape{self.Y_train}\n,X_test shape{self.X_test}\n,Y_test shape {self.Y_test}\n, Classes is: {self.classes}\n, number of training examples= {self.X_train.shape[0]}\n,number of test examples= {self.X_test.shape[0]}'

    def __repr__(self):
        return f'{self.X_train}{self.Y_train}{self.X_test}{self.Y_test}{self.classes}{self.X_train.shape[0]}{self.X_test.shape[0]}'



