from  one_hot import convert_to_one_hot
class NoramlPic():
    def __init__(self, myobj):
        try:
            self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes =  myobj
            #print(self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes)
        except Exception as err:
            print(err)
    def norm(self):
        self.X_train=self.X_train/255
        self.X_test=self.X_test/255
        self.Y_train=convert_to_one_hot(self.Y_train,len(self.classes)).T
        self.Y_test = convert_to_one_hot(self.Y_test, len(self.classes)).T
        return self.X_train,self.X_test,self.Y_train,self.Y_test,self.classes

    def __str__(self):
        return f'Dataset inclueds: X_train shape {self.X_train.shape}\nY_train shape{self.Y_train.shape}\n,X_test shape{self.X_test.shape}\n,Y_test shape {self.Y_test.shape}\n, Classes is: {self.classes}\n, number of training examples= {self.X_train.shape[0]}\n,number of test examples= {self.X_test.shape[0]}'

    def __repr__(self):
        return f'{self.X_train}{self.Y_train}{self.X_test}{self.Y_test}{self.classes}{self.X_train.shape[0]}{self.X_test.shape[0]}'



