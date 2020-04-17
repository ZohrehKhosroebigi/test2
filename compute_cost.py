import tensorflow as tf
class Costfunc():
    def cost(self,Z,Y):
        self.c=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))
        return self.c

    def __str__(self):
        try:
            return f'Dataset inclueds: X_train_orig shape {self.c}'
        except Exception as err:
            print(err)

    def __repr__(self):
        try:
            return f' {self.c}'
        except Exception as err:
            print(err)
