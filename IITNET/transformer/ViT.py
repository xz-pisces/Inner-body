from keras.layers import Input, Conv2D, Reshape, Concatenate, add, Dropout, Dense, Lambda,Permute,RepeatVector
from transformer.MSA import MultiHeadAttention, FeedForwardNetwork, gelu
from transformer.transformer import positional_embedding
from transformer.LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Add, Conv2D, Input, Lambda, Activation, Conv2DTranspose
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization, Multiply
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax, MaxPool2D

def encoder_block(x, hidden_dim=128, att_drop_rate=0., num_heads=4, mlp_dim=64, drop_rate=0.1):
    # MSA



    inpt = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(hidden_dim, num_heads)([x,x,x])   # self-attention
    x = Dropout(drop_rate)(x)
    x = add([inpt, x])
    # layer norm

    # FFN
    inpt = x
    out_dim = K.int_shape(x)[-1]
    x = LayerNormalization()(x)
    x = FeedForwardNetwork(mlp_dim, out_dim, activation=gelu, drop_rate=drop_rate)(x)
    x = add([inpt, x])
    # x = LayerNormalization(x)

    return x

def SelfAttention(x, channels):
    x_shape = K.shape(x)
    # self attention
    f = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
    f = MaxPool2D(pool_size=(2, 2))(f)
    g = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
    # flatten hw * (1/8)c matmul (1/8)c * (1/4)hw -> hw * (1/4)hw
    shape = (K.shape(g)[0], -1, K.shape(g)[-1])
    g = Lambda(tf.reshape, arguments={'shape': shape})(g)
    shape = (K.shape(f)[0], -1, K.shape(f)[-1])
    f = Lambda(tf.reshape, arguments={'shape': shape})(f)
    s = Lambda(tf.matmul, arguments={'b': f, 'transpose_b': True})(g)
    # attention map
    beta = Softmax()(s)
    h = Conv2D(filters=channels // 2, kernel_size=1, strides=1, padding='same')(x)
    h = MaxPool2D(pool_size=(2, 2))(h)
    shape = (K.shape(h)[0], -1, K.shape(h)[-1])
    h = Lambda(tf.reshape, arguments={'shape': shape})(h)
    # hw * (1/4)hw matmul (1/4)hw * (1/2)c -> hw * (1/2)c
    o = Lambda(tf.matmul, arguments={'b': h, 'transpose_b': False})(beta)
    # gamma
    gamma = K.variable(0.0)
    shape = (x_shape[0], x_shape[1], x_shape[2], channels // 2)
    o = Lambda(tf.reshape, arguments={'shape': shape})(o)
    # xch * scale ** 2
    o = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(o)
    o = Lambda(tf.multiply, arguments={'y': gamma})(o)
    x = Add()([o, x])
    return x
#
# def SelfAttention(x, channels):
#     channels =128
#
#     f = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
#     f = MaxPool2D(pool_size=(2, 2))(f)
#     g = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
#     g = Reshape((196,16))(g)
#
#     f = Reshape((49,16))(f)
#     f=  Permute((2,1))(f)
#     s = Lambda(lambda x: K.batch_dot(*x))([g,f])
#
#     beta = Softmax()(s)
#     h = Conv2D(filters=channels // 2, kernel_size=1, strides=1, padding='same')(x)
#     h = MaxPool2D(pool_size=(2, 2))(h)
#     h =Reshape((49,64))(h)
#     o= Lambda(lambda x: K.batch_dot(*x))([beta,h])
#     gamma = K.constant(0, shape=(14,14,128))
#     gamma = Lambda(lambda x:  tf.expand_dims(gamma, 0))(gamma)
#     gamma = Lambda(lambda x: tf.tile(gamma, [tf.shape(o)[0],1, 1, 1]))(gamma)
#
#     o =Reshape((14,14,64))(o)
#     o = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(o)
#     o = Multiply()([o,gamma])
#     x = Add()([o, x])
#     return x

# def SelfAttention(x, channels):
#     channels =128
#     # x_shape = K.shape(x)
#
#     # self attention
#     f = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
#     f = MaxPool2D(pool_size=(2, 2))(f)
#     g = Conv2D(filters=channels // 8, kernel_size=1, strides=1, padding='same')(x)
#     # flatten hw * (1/8)c matmul (1/8)c * (1/4)hw -> hw * (1/4)hw
#     # shape = (K.shape(g)[0], -1, K.shape(g)[-1])
#     # g = Lambda(tf.reshape, arguments={'shape': shape})(g)
#     g =Reshape((196,16))(g)
#
#     # shape = (K.shape(f)[0], -1, K.shape(f)[-1])
#     # f = Lambda(tf.reshape, arguments={'shape': shape})(f)
#     f =Reshape((49,16))(f)
#     # print("ss")
#     # s = Lambda(tf.matmul, arguments={'b': f, 'transpose_b': True})(g)
#
#     f= Permute((2,1))(f)
#     s = Lambda(lambda x: K.batch_dot(*x))([g,f])
#
#
#
#     # attention map
#     # beta = Softmax()(s)
#
#     beta = Lambda(lambda x: Softmax()(s))(s)
#     h = Conv2D(filters=channels // 2, kernel_size=1, strides=1, padding='same')(x)
#
#
#     h = Lambda(lambda x: MaxPool2D(pool_size=(2, 2))(h))(h)
#     # h = MaxPool2D(pool_size=(2, 2))(h)
#     # shape = (K.shape(h)[0], -1, K.shape(h)[-1])
#     # h = Lambda(tf.reshape, arguments={'shape': shape})(h)
#     h =Reshape((49,64))(h)
#     # hw * (1/4)hw matmul (1/4)hw * (1/2)c -> hw * (1/2)c
#     # o = Lambda(tf.matmul, arguments={'b': h, 'transpose_b': False})(beta)
#
#     o= Lambda(lambda x: K.batch_dot(*x))([beta,h])
#
#     # gamma
#     # gamma = K.variable(0.0)
#     # shape = (x_shape[0], x_shape[1], x_shape[2], channels // 2)
#     gamma = K.constant(0, shape=(14,14,128))
#     # gamma =Reshape(())
#     # gamma = Lambda(lambda x:  tf.expand_dims(gamma, 0))(gamma)
#     gamma = Lambda(lambda x: tf.tile(gamma, [tf.shape(o)[0],1, 1, 1]))(gamma)
#
#     # gamma =Reshape((14,14,128))(gamma)
#     # gamma = keras.initializers.Zeros(shape=(14,14,128))
#     # gamma = K.variable(value=gamma)
#     # o = Lambda(tf.reshape, arguments={'shape': shape})(o)
#     o =Reshape((14,14,64))(o)
#     # xch * scale ** 2
#     o = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(o)
#     # o = Lambda(tf.multiply, arguments={'y': gamma})(o)
#     # gamma = Reshape((1,1,1))(gamma)
#     # gamma = RepeatVector(1)(gamma)
#     o = Multiply()([o,gamma])
#     x = Add()([o, x])
#     # x =Reshape((14,14,128))(x)
#     return x















