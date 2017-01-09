from keras.models import Model
from keras.layers.core import Activation, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from layers import SubPixelUpscaling
import json

class Tiramisu():



    def __init__(self, nb_classes, img_dim, nb_dense_block=5, growth_rate=12, nb_filter=16, nb_layers=4, upsampling_conv=128,
                            bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4, upscaling_type='deconv',
                            verbose=True):

        self.nb_classes=nb_classes
        self.img_dim=img_dim
        self.nb_dense_block=nb_dense_block
        self.growth_rate=growth_rate
        self.nb_filter=nb_filter
        self.nb_layers=nb_layers
        self.upsampling_conv=upsampling_conv    
        self.bottleneck=bottleneck
        self.reduction=reduction
        self.dropout_rate=dropout_rate
        self.weight_decay=weight_decay
        self.upscaling_type=upscaling_type
        self.verbose=verbose
        self.create()

    def conv_block(self, input_tensor, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, Relu 3x3, Conv2D, optional bottleneck block and dropout

        Args:
            input_tensor: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor

        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)

        '''

        concat_axis = 1

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(input_tensor)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                              W_regularizer=l2(weight_decay))(x)

            if dropout_rate:
                x = Dropout(dropout_rate)(x)

            x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
            x = Activation('relu')(x)

        x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                          W_regularizer=l2(weight_decay))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x


    def transition_down_block(self, input_tensor, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D

        Args:
            input_tensor: keras tensor
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor

        Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

        '''

        concat_axis = 1

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(input_tensor)
        x = Activation('relu')(x)
        x = Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                          W_regularizer=l2(weight_decay))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x


    def transition_up_block(self, input_tensor, nb_filters, type='deconv', output_shape=None, weight_decay=1E-4):
        ''' deconv Upscaling (factor = 2)

        Args:
            input_tensor: keras tensor
            nb_filters: number of layers
            type:'deconv'. Determines type of upsampling performed
            output_shape: required if type = 'deconv'. Output shape of tensor
            weight_decay: weight decay factor

        Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

        '''


        x = Deconvolution2D(nb_filters, 3, 3, output_shape, activation='relu', border_mode='same',
                            subsample=(2, 2))(input_tensor)



    def dense_block(self, x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor

        Returns: keras tensor with nb_layers of conv_block appended

        '''

        concat_axis = 1

        feature_list = [x]

        for i in range(nb_layers):
            x = self.conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            feature_list.append(x)
            x = merge(feature_list, mode='concat', concat_axis=concat_axis)
            nb_filter += growth_rate

        return x, nb_filter


    def create(self):
        ''' Build the create_dense_net model

        Args:
            nb_classes: Number of classes
            img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Setting -1 indicates initial number of filters is 2 * growth_rate
            nb_layers: number of layers in each dense block. Can be an -1, a positive integer or a list

                       If -1, it computes the nb_layer from depth

                       If positive integer, a set number of layers per dense block

                       If list, nb_layer is used as provided.
                       Note that list size must be (nb_dense_block + 1)

            upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay
            upscaling_type: method of upscaling. Can be 'subpixel' or 'deconv'
            verbose: print the model type

        Returns: keras tensor with nb_layers of conv_block appended

        '''

        batch_size = None

        model_input = Input(shape=self.img_dim)

        concat_axis = 1 

        _, rows, cols = self.img_dim


        if self.reduction != 0.0:
            assert self.reduction <= 1.0 and self.reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

        # check if upsampling_conv has minimum number of filters
        # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
        assert self.upsampling_conv > 12 and self.upsampling_conv % 4 == 0, "upsampling_conv number of channels must " \
                                                                 "be a positive number divisible by 4 and greater " \
                                                                  "than 12"



        # layers in each dense block
        if type(self.nb_layers) is list or type(self.nb_layers) is tuple:
            self.nb_layers = list(nb_layers) # Convert tuple to list

            assert len(self.nb_layers) == (self.nb_dense_block + 1), "If list, nb_layer is used as provided. " \
                                                            "Note that list size must be (nb_dense_block + 1)"

            final_nb_layer = self.nb_layers[-1]
            self.nb_layers = self.nb_layers[:-1]

        else:
            final_nb_layer = self.nb_layers
            self.nb_layers = [self.nb_layers] * self.nb_dense_block

        if self.bottleneck:
            self.nb_layers = [int(layer // 2) for layer in self.nb_layers]

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if self.nb_filter <= 0:
            self.nb_filter = 2 * self.growth_rate

        # compute compression factor
        compression = 1.0 - self.reduction

        # Initial convolution
        x = Convolution2D(48, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                          W_regularizer=l2(self.weight_decay))(model_input)

        skinput_tensor_connection = x
        skinput_tensor_list = []

        # Add dense blocks and transition down block
        for block_idx in range(self.nb_dense_block):
            x, nb_filter = self.dense_block(x, self.nb_layers[block_idx], self.nb_filter, self.growth_rate, bottleneck=self.bottleneck,
                                       dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)

            # Skinput_tensor connection
            x = merge([x, skinput_tensor_connection], mode='concat', concat_axis=concat_axis)
            skinput_tensor_list.append(x)

            # add transition_block
            x = self.transition_down_block(x, nb_filter, compression=compression, dropout_rate=self.dropout_rate,
                                      weight_decay=self.weight_decay)
            nb_filter = int(nb_filter * compression)

            # Preserve transition for next skinput_tensor connection after dense
            skinput_tensor_connection = x

        # The last dense_block does not have a transition_down_block
        x, nb_filter = self.dense_block(x, final_nb_layer, nb_filter, self.growth_rate, bottleneck=self.bottleneck,
                                   dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)

        out_shape = [batch_size, nb_filter, rows // 16, cols // 16]
        
        # Add dense blocks and transition up block
        for block_idx in range(self.nb_dense_block):
            x = self.transition_up_block(x, nb_filters=self.upsampling_conv, type=self.upscaling_type, output_shape=out_shape)

            out_shape[2] *= 2
            out_shape[3] *= 2


            x = merge([x, skinput_tensor_list.pop()], mode='concat', concat_axis=concat_axis)

            x, nb_filter = self.dense_block(x, self.nb_layers[-block_idx], nb_filter, growth_rate, bottleneck=self.bottleneck,
                                       dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)

        x = Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(self.weight_decay),
                          bias=False)(x)

        channel, row, col = self.img_dim


        x = Reshape((row * col, self.nb_classes))(x)

        x = Activation('softmax')(x)

        densenet = Model(input=model_input, output=x, name="create_dense_net")

        # Compute depth
        nb_conv_layers = len([layer.name for layer in densenet.layers
                              if layer.__class__.__name__ == 'Convolution2D'])

        depth = nb_conv_layers -  self.nb_dense_block # For 1 extra convolution layers per transition up

        if self.verbose: print('Total number of convolutions', depth)

        if self.verbose:
            if self.bottleneck and not self.reduction:
                print("Bottleneck DenseNet-B-%d-%d created." % (depth, self.growth_rate))
            elif not self.bottleneck and self.reduction > 0.0:
                print("DenseNet-C-%d-%d with %0.1f compression created." % (depth, self.growth_rate, compression))
            elif self.bottleneck and self.reduction > 0.0:
                print("Bottleneck DenseNet-BC-%d-%d with %0.1f compression created." % (depth, self.growth_rate, compression))
            else:
                print("DenseNet-%d-%d created." % (depth, self.growth_rate))

        return densenet



nb_layers = [4, 5, 7, 10, 12, 15]
model = Tiramisu(nb_classes=12,img_dim=(3, 224, 224), nb_layers=nb_layers)
model.summary()

with open('tiramisu_fc_dense103_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=3))