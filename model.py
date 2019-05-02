from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2




def extract_faetures(input_layer, filter_size):
    # =============================================================================
    #     This function performs a multi-scale convolutional block
    # =============================================================================

    conv_layer_1 = Conv1D(filters=filter_size[0], kernel_size=8, strides=4, kernel_initializer='RandomNormal',
                          activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                          bias_initializer='RandomNormal')(input_layer)
    conv_layer_2 = Conv1D(filters=filter_size[1], kernel_size=12, strides=4, kernel_initializer='RandomNormal',
                          activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                          bias_initializer='RandomNormal')(input_layer)
    conv_layer_3 = Conv1D(filters=filter_size[2], kernel_size=24, strides=4, kernel_initializer='RandomNormal',
                          activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                          bias_initializer='RandomNormal')(input_layer)

    pool_1 = GlobalMaxPooling1D()(conv_layer_1)
    pool_2 = GlobalMaxPooling1D()(conv_layer_2)
    pool_3 = GlobalMaxPooling1D()(conv_layer_3)

    concate = concatenate([pool_1, pool_2, pool_3])

    return concate


def model(input_shape, filter_size, opt_func, lr, fc):
    input_layer = Input(shape=input_shape)

    flat1 = extract_faetures(input_layer, filter_size)

    hidden = Dense(fc)(flat1)
    hidden = Activation('relu')(hidden)
    output = Dense(1)(hidden)
    output = Activation('sigmoid')(output)
    model = Model(inputs=input_layer, outputs=output)


    if 'sgd' in opt_func:
        opt = optimizers.SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

