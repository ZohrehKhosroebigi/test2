import tensorflow as tf
class InitParams():
    params={}
    def init_params(self,w_1,w1,dim1):
        w_1 = tf.get_variable(w1, dim1, initializer=tf.contrib.layers.xavier_initializer())
        self.__class__.params[w1]= w_1
        print("000000000000000000"+str(self.__class__.params[w1]))
        return self.__class__.params
    def __str__(self):
        return f'initilized params are {self.__class__.params}\n'
    def __repr__(self):
        return f'{self.__class__.params.items}'