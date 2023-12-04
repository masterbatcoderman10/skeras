import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

#custom keras layer
class ConvStack(layers.Layer):

    def __init__(self, filters, kernel_size, padding='same', activation='relu', **kwargs):
        super(ConvStack, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.conv_stack = K.Sequential()
        for f in filters:
            self.conv_stack.add(
                layers.Conv2D(f, kernel_size, padding=padding, activation=activation),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
            )
        
    def call(self, inputs):

        return self.conv_stack(inputs)


