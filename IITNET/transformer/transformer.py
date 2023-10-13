from keras.layers import Dropout, Embedding, Input
# from keras_layer_normalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
from transformer.MSA import MultiHeadAttention, FeedForwardNetwork
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Average, Concatenate, Add, Reshape,Dropout,add


# encoder: stacks of EncoderBlock
class Encoder(Model):
    def __init__(self, vocab_size, maxlen, num_layers=6, model_dim=512,
                 num_heads=8, hidden_dim=128, rate=0.1):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, model_dim)
        self.pos_embedding = positional_embedding(maxlen, model_dim)  # tensor

        self.dropout = Dropout(rate)
        self.encoder_layers = []
        for i in range(num_layers):
            self.en = EncoderBlock(model_dim, num_heads, hidden_dim, rate)   # must use self scope to keep active
            self.encoder_layers.append(self.en)

    def call(self, x, training=None, mask=None):
        # input embedding + positional embedding
        x = self.embedding(x) + self.pos_embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        return x

    def compute_output_shape(self, input_shape):
        B, N = input_shape
        return (B,N,self.model_dim)


# decoder: stacks of DecoderBlock
class Decoder(Model):
    def __init__(self, vocab_size, maxlen, num_layers=6, model_dim=512,
                 num_heads=8, hidden_dim=128, rate=0.1):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, model_dim)
        self.pos_embedding = positional_embedding(maxlen, model_dim)  # tensor

        self.dropout = Dropout(rate)
        self.decoder_layers = []
        for i in range(num_layers):
            self.de = DecoderBlock(model_dim, num_heads, hidden_dim, rate)
            self.decoder_layers.append(self.de)

    def call(self, inputs, training=None):
        x, enc, look_ahead_mask = inputs
        # input embedding + positional embedding
        x = self.embedding(x) + self.pos_embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i]([x, enc, look_ahead_mask], training=training)

        return x

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[1]
        return (B,N,self.model_dim)


# encoder block
class EncoderBlock(Model):
    def __init__(self, model_dim=512, num_heads=8, hidden_dim=1024, rate=0.1):
        super(EncoderBlock, self).__init__()

        self.msa = MultiHeadAttention(model_dim, num_heads)
        self.ffn = FeedForwardNetwork(hidden_dim, model_dim)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.model_dim = model_dim

    def call(self, x, training=None):
        # multi head attention
        attn_output = self.msa(inputs=[x, x, x])
        attn_output = self.dropout1(attn_output, training=training)
        # residual: add & layernorm
        out1 = self.layernorm1(x + attn_output)
        # ffn layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # residual: add & layernorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape
        return (B,N,self.model_dim)


class DecoderBlock(Model):
    def __init__(self, model_dim=512, num_heads=8, hidden_dim=1024, rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mmsa = MultiHeadAttention(model_dim, num_heads)   # masked msa
        self.msa = MultiHeadAttention(model_dim, num_heads)
        self.ffn = FeedForwardNetwork(hidden_dim, model_dim)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        self.model_dim = model_dim

    def call(self, inputs, training=None):
        x, enc, look_ahead_mask = inputs
        # masked multi head attention
        attn_output = self.mmsa(inputs=[x, x, x], mask=look_ahead_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # multi head attention
        attn_output = self.msa(inputs=[out1, enc, enc])
        attn_output = self.dropout2(attn_output, training=training)
        out2 = self.layernorm2(out1 + attn_output)
        # ffn layer
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm2(out2 + ffn_output)

        return out3

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[1]
        return (B,N,self.model_dim)


def positional_embedding(seq_len, model_dim):
    PE = np.zeros((seq_len, model_dim))
    # PE = pe
    for i in range(seq_len):
        for j in range(model_dim):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_dim))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_dim))
    PE = K.constant(np.expand_dims(PE, axis=0))
    # b, sq_len, input_dim = K.int_shape(inter)
    # PE = tf.tile(PE, (-1, 1, 1))
    # PE = K.constant(PE)
    # PE = Reshape((1,196,128))(PE)
    # PE = tf.tile(tf.expand_dims(PE, 0), (batch_size,1,1))
    return PE


if __name__ == '__main__':

    # test encoder block
    # x = Input((20,10))
    # y = EncoderBlock(model_dim=10, num_heads=2, hidden_dim=16, rate=0.1)(x, training=True)
    # y = EncoderBlock(model_dim=10, num_heads=2, hidden_dim=16, rate=0.1)(y, training=True)

    # test encoder
    # x = Input((20,))
    # y = Encoder(vocab_size=10, maxlen=20, num_layers=2, model_dim=10, num_heads=1, hidden_dim=16, rate=0.1)(x, training=True)

    # model = Model(x,y)
    # model.summary()

    # test decoder block
    enc = Input((20,10))
    mask = Input((20,20))
    # x = Input((20,10))
    # y = DecoderBlock(model_dim=10, num_heads=2, hidden_dim=16, rate=0.1)([x,enc,mask])
    # y = DecoderBlock(model_dim=10, num_heads=2, hidden_dim=16, rate=0.1)([y,enc,mask])

    # # test decoder
    x = Input((20,))
    y = Decoder(vocab_size=10, maxlen=20, num_layers=2, model_dim=10, num_heads=2, hidden_dim=16, rate=0.1)([x,enc,mask])

    model = Model([x,enc,mask],y)
    model.summary()



