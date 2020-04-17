from init_params import InitParams
import tensorflow as tf
class ForwardPorpagation():
    layer={}

    def forward(self,X,W,c_s,c_pad,p_size,p_s,p_pad):
        ForwardPorpagation.layer["Z1"]=tf.nn.conv2d(X,W,strides=c_s,padding=c_pad)
        print("Done for ----------------"+str(ForwardPorpagation.layer["Z1"]))
        ForwardPorpagation.layer["A1"]=tf.nn.relu(ForwardPorpagation.layer["Z1"])
        print("Done for ----------------"+str(ForwardPorpagation.layer["A1"]))
        ForwardPorpagation.layer["P1"]=tf.nn.max_pool(ForwardPorpagation.layer["A1"],ksize=p_size,strides=p_s,padding=p_pad)
        print("Done for ----------------"+str(ForwardPorpagation.layer["P1"]))
        print("Done for ----------------")
        return ForwardPorpagation.layer

    def full_connected(self,classes):
        F=tf.contrib.layers.flatten(ForwardPorpagation.layer["P1"])
        ForwardPorpagation.layer["Z1"]=tf.contrib.layers.fully_connected(F,classes, activation_fn=None)
        return ForwardPorpagation.layer
    def __str__(self):
        return f'Forward is {ForwardPorpagation.layer["Z1"]} \n'
    def __repr__(self):
        return f'{ForwardPorpagation.layer["Z1"]}'


