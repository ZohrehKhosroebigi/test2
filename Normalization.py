from  one_hot import convert_to_one_hot
import tensorflow as tf
class NoramlPic():

    X_train=""
    Y_train=""
    X_test=""
    Y_test=""
    classes =""
    def __init__(self, myobj):
        try:
            #self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes=myobj
            NoramlPic.X_train, NoramlPic.Y_train, NoramlPic.X_test, NoramlPic.Y_test, NoramlPic.classes =  myobj
            #print(self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes)
        except Exception as err:
            print(err)
    def norm(self):

        NoramlPic.X_train=NoramlPic.X_train/255.
        #NoramlPic.X_train = tf.convert_to_tensor(NoramlPic.X_train, dtype=tf.int32)
        NoramlPic.X_test=NoramlPic.X_test/255.
        #NoramlPic.X_test = tf.convert_to_tensor(NoramlPic.X_test, dtype=tf.int32)
        NoramlPic.Y_train=convert_to_one_hot(NoramlPic.Y_train,len(NoramlPic.classes)).T
        NoramlPic.Y_test = convert_to_one_hot(NoramlPic.Y_test, len(NoramlPic.classes)).T
        #print("################",str(len(self.classes)))
        print("number of training examples = " + str(NoramlPic.X_train.shape[0]))
        print("number of test examples = " + str(NoramlPic.X_test.shape[0]))
        print("X_train shape: " + str(NoramlPic.X_train.shape))
        print("Y_train shape: " + str(NoramlPic.Y_train.shape))
        print("X_test shape: " + str(NoramlPic.X_test.shape))
        print("Y_test shape: " + str(NoramlPic.Y_train.shape))
        NoramlPic.classes=self.classes
        return NoramlPic.X_train,NoramlPic.X_test,NoramlPic.Y_train,NoramlPic.Y_test,NoramlPic.classes

    def __str__(self):
        return f'Dataset inclueds: X_train shape {NoramlPic.X_train}\nY_train shape{NoramlPic.Y_train}\n,X_test shape{NoramlPic.X_test}\n,Y_test shape {NoramlPic.Y_test}\n, Classes is: {NoramlPic.classes}\n, number of training examples= {NoramlPic.X_train.shape[0]}\n,number of test examples= {NoramlPic.X_test.shape[0]}'
    def __repr__(self):
        return f'{NoramlPic.X_train}{NoramlPic.Y_train}{NoramlPic.X_test}{NoramlPic.Y_test}{NoramlPic.classes}{NoramlPic.X_train.shape[0]}{NoramlPic.X_test.shape[0]}'



