from keras.layers import Layer
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras import constraints


class LayerNormalization(Layer):

    # given inputs: [b,(hwd),c], for each sample, compute norm over (hwdc)-dims
    # currently only available for 1D-tensor

    def __init__(self,
                 rescale=True,
                 epsilon=1e-5,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        if epsilon is None:
            epsilon = K.epsilon()
        self.rescale=rescale
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        # rescale factor, for each sample, broadcast from last-dim
        shape = input_shape[1:]
        if self.rescale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=[1,2], keepdims=True)
        # norm
        variance = K.var(inputs, axis=[1,2], keepdims=True)
        outputs = (inputs - mean) / K.sqrt(variance + self.epsilon)
        # rescale
        outputs = self.gamma*outputs + self.beta
        return outputs

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model

    x = Input((128,128,16))
    y = LayerNormalization()(x)

    model = Model(x,y)
    model.summary()

    print(y)
