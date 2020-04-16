import tensorflow as tf
class CreatePlaceholder():
    #creare place holder for input that will be fed into the model
    #argumants of this class will be scalar
    def createplaceholder(self,n_h,n_w,n_c,n_y):
        try:
            self.X=tf.placeholder(tf.float32,[None,n_h,n_w,n_c])
            self.Y=tf.placeholder(tf.float32,[None,n_y])
            return self.X,self.Y
        except Exception as err:
            print(err)
    def __str__(self):
        return f'place holders are:  {self.X}\n{self.Y}'

    def __repr__(self):
        return f'{self.X}{self.Y}'


