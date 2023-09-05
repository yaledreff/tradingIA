from tensorflow.keras.layers import Layer, Dense, Attention, concatenate, Permute, Reshape, RepeatVector, Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import *
#from tensorflow.keras.layers.core import *

def attention_3d_block(inputs, singleAttentionVector = True):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    timesteps = int(inputs.shape[1])
    print(inputs.shape)
    a = Permute((2, 1))(inputs) # From (batch_size, time_steps, input_dim) to (batch_size, input_dim ,time_steps)
    #a = Reshape((input_dim, timesteps))(a), 
    a = Dense(timesteps, activation='softmax')(a)
    if singleAttentionVector: # if True, the attention vector is shared across the input_dimensions where the attention is applied.
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = concatenate([inputs, a_probs])
    #return output_attention_mul
    return a_probs

# Add attention layer to the deep learning network
class attention1(Layer):
    def __init__(self,**kwargs):
        super(attention1,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention1, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

# Define the Attention layer
class attention2(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        # Compute attention scores
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector