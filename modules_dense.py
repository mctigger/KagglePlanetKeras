from keras.layers import Conv2D, merge, BatchNormalization, SpatialDropout2D, Activation


def _conv_bottleneck(filters, bottleneck_filters, activation, dropout_rate, use_bias, bn_scale, kernel_regularizer, kernel_initializer):
    def call(x):
        x = BatchNormalization(scale=bn_scale)(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(bottleneck_filters, (1, 1), kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, padding='same', use_bias=use_bias)(x)
        x = BatchNormalization(scale=bn_scale)(x)
        x = SpatialDropout2D(dropout_rate)(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters, (3, 3), kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, padding='same', use_bias=use_bias)(x)

        return x

    return call


def DenseBottleneckBlock(
        depth,
        k,
        dropout_rate=0.0,
        kernel_regularizer=None,
        kernel_initializer='he_normal',
        activation='elu',
        use_bias=False,
        bn_scale=False):

    def call(x):
        layer_outputs = [x]
        for i in range(depth):
            bottleneck_filters = 4*k
            x = _conv_bottleneck(k, bottleneck_filters, activation, dropout_rate, use_bias, bn_scale, kernel_regularizer, kernel_initializer)(x)

            layer_outputs.append(x)
            x = merge(layer_outputs, mode='concat')

        return x

    return call


def CompressionBlock(filters, activation='elu', use_bias=False, bn_scale=False, kernel_regularizer=None, kernel_initializer='he_normal', name=None):
    bn_name = None
    conv_name = None

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'

    def call(x):
        x = BatchNormalization(scale=bn_scale, name=bn_name)(x)
        x = Activation(activation)(x)
        x = Conv2D(
            filters,
            (1, 1),
            padding='same',
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name=conv_name)(x)

        return x

    return call


def _conv(filters, activation, dropout_rate, use_bias, bn_scale, kernel_regularizer):
    def call(x):
        x = BatchNormalization(scale=bn_scale)(x)
        x = SpatialDropout2D(dropout_rate)(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters, (3, 3), padding='same', use_bias=use_bias, kernel_regularizer=kernel_regularizer)(x)

        return x

    return call


def DenseBlock(depth, k=12, dropout_rate=0, activation='elu', use_bias=False, bn_scale=False, kernel_regularizer=None):
    def call(x):
        layer_outputs = [x]
        for i in range(depth):
            x = _conv(k, activation, dropout_rate, use_bias, bn_scale, kernel_regularizer)(x)

            layer_outputs.append(x)
            x = merge(layer_outputs, mode='concat')

        return x

    return call
