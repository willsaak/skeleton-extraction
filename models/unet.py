from tensorflow import keras


def double_conv2d(x, filters, kernel_size, padding='same', batch_normalization=False, dropout_rate=0.):
    """
    Building block of U-Net architecture: two successive identical convolutions + batch normalization and dropout
    afterwards. Parameters `x`, `filters`, `kernel_size` and `padding` are the same as in the regular Keras Conv2D
    layer and valid for both convolutions.
    :param x: [4D tensor] input
    :param filters: [integer] number of filters or number of channels in the output
    :param kernel_size: [integer/tuple] size of convolution kernels
    :param padding: ['valid'/'same'] convolution padding
    :param batch_normalization: [bool] whether to apply optional batch normalization between convolution and ReLU
    :param dropout_rate: [float] optional dropout rate (dropout is not applied if is equal to 0.0)
    :return: [4D tensor] output
    """
    use_bias = not batch_normalization
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    x = keras.layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias, kernel_initializer='he_normal')(x)
    if batch_normalization is True:
        x = keras.layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias, kernel_initializer='he_normal')(x)
    if batch_normalization is True:
        x = keras.layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = keras.layers.Activation('relu')(x)
    if dropout_rate > 0.:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def unet(weights=None,
         input_shape=(256, 256, 1),
         classes=1,
         background_as_class=False,
         up_conv='upsampling',
         batch_normalization=False,
         dropout_rate=(0., 0., 0., .5, .5, 0., 0., 0., 0.)) -> keras.models.Model:
    """
    Instantiates the U-Net architecture for Keras.
    Note that convolutions have `padding='same'` argument instead of `padding='valid'` in reference paper in order to
    have the same input and output shapes. `padding='valid'` is currently not supported.
    Since U-Net is fully-convolutional network, in fact it requires no input height and width, but just the number of
    channels. However, it can be useful to set input height and width to obtain feature map shapes in `model.summary()`
    Using up-sampling with convolution afterwards instead of transpose convolution can avoid some undesirable artifacts
    (see e.g. https://distill.pub/2016/deconv-checkerboard/). Bilinear interpolation was chosen as more accurate and
    the same kernel size as in transpose convolution is used in order to not to increase the number of parameters
    Dropout rates were discussed only briefly in reference paper, default values were therefore taken from the
    reference Caffe implementation.
    :param weights: optional path to the weights file to be loaded (random initialization if `None`)
    :param input_shape: [integer/tuple] optional number of input channels or input shape
    :param classes: [int] optional number of classes to predict
    :param background_as_class: [bool] whether to create additional channel for background class
    :param up_conv: ['deconvolution'/'upsampling'] how to perform the up-convolution
    :param batch_normalization: [bool] whether to apply batch normalization after each convolution
    :param dropout_rate: [integer/tuple/list] dropout rate to apply to all building blocks or
                         tuple/list of size 9 with block-wise dropout rates
    :return: Keras model instance
    """

    if isinstance(input_shape, int):
        if keras.backend.image_data_format() == 'channels_last':
            input_shape = (None, None, input_shape)
        else:
            input_shape = (input_shape, None, None)
    elif isinstance(input_shape, tuple) and len(input_shape) == 3:
        if keras.backend.image_data_format() == 'channels_last':
            input_height, input_width = input_shape[0], input_shape[1]
        else:
            input_height, input_width = input_shape[1], input_shape[2]
        if input_height % 16 != 0 or input_width % 16 != 0:
            raise ValueError("Input height and width should be a multiply of 16 in order to do 4 down-samplings and "
                             "then 4 up-samplings correctly")
    else:
        raise ValueError("The `input_shape` argument should be either integer (number of input channels)"
                         "or tuple of size 3 with input shape")

    if background_as_class is True:
        # Add one more class for background
        classes += 1
        # Classes (and background) probabilities in each pixel are conditional dependent
        top_activation = 'softmax'
    else:
        # Classes (and background) probabilities in each pixel are independent
        # Some pixel is background if all classes activations in this pixel are nearly zeros
        top_activation = 'sigmoid'

    if up_conv not in ('deconvolution', 'upsampling'):
        raise ValueError("The `up_conv` argument should be either 'deconvolution' (up-convolution by transposed"
                         "convolution or so called deconvolution) or 'upsampling' (up-convolution by up-sampling and"
                         "regular convolution")

    if isinstance(dropout_rate, float):
        dropout_rate = [dropout_rate] * 9
    elif not isinstance(dropout_rate, tuple) and not isinstance(dropout_rate, list) or len(dropout_rate) != 9:
        raise ValueError("The `dropout_rate` argument should be either float (the same dropout rate"
                         "for all building blocks) or list/tuple of size 9 with block-wise dropout rates")

    channel_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    data = keras.layers.Input(input_shape)

    down0 = double_conv2d(data, 64, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[0])

    down1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(down0)
    down1 = double_conv2d(down1, 128, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[1])

    down2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(down1)
    down2 = double_conv2d(down2, 256, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[2])

    down3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(down2)
    down3 = double_conv2d(down3, 512, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[3])

    down4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(down3)
    down4 = double_conv2d(down4, 1024, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[4])

    if up_conv == 'deconvolution':
        up3 = keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), kernel_initializer='he_normal')(down4)
    else:
        up3 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(down4)
        up3 = keras.layers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(up3)
    up3 = keras.layers.Activation('relu')(up3)
    up3 = keras.layers.concatenate([down3, up3], axis=channel_axis)
    up3 = double_conv2d(up3, 512, 3, padding='same',
                        batch_normalization=batch_normalization, dropout_rate=dropout_rate[5])

    if up_conv == 'deconvolution':
        up2 = keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), kernel_initializer='he_normal')(up3)
    else:
        up2 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(up3)
        up2 = keras.layers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(up2)
    up2 = keras.layers.Activation('relu')(up2)
    up2 = keras.layers.concatenate([down2, up2], axis=channel_axis)
    up2 = double_conv2d(up2, 256, 3, padding='same',
                        batch_normalization=batch_normalization, dropout_rate=dropout_rate[6])

    if up_conv == 'deconvolution':
        up1 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), kernel_initializer='he_normal')(up2)
    else:
        up1 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(up2)
        up1 = keras.layers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up1)
    up1 = keras.layers.Activation('relu')(up1)
    up1 = keras.layers.concatenate([down1, up1], axis=channel_axis)
    up1 = double_conv2d(up1, 128, 3, padding='same',
                        batch_normalization=batch_normalization, dropout_rate=dropout_rate[7])

    if up_conv == 'deconvolution':
        up0 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), kernel_initializer='he_normal')(up1)
    else:
        up0 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(up1)
        up0 = keras.layers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up0)
    up0 = keras.layers.Activation('relu')(up0)
    up0 = keras.layers.concatenate([down0, up0], axis=channel_axis)
    up0 = double_conv2d(up0, 64, 3, padding='same',
                        batch_normalization=batch_normalization, dropout_rate=dropout_rate[8])

    score = keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(up0)
    score = keras.layers.Activation(top_activation)(score)

    model = keras.models.Model(data, score)

    if weights is not None:
        model.load_weights(weights)

    return model
