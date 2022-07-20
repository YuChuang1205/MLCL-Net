def MLCLNet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = ReLU()(conv1_1)


    conv1_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = ReLU()(conv1_2)   

    
    #layer2_1
    conv2_1_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    conv2_1_1 = BatchNormalization()(conv2_1_1)
    conv2_1_1 = ReLU()(conv2_1_1)

    conv2_1_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_1_1)
    conv2_1_2 = BatchNormalization()(conv2_1_2)


    con_conv2_1 = Add()([conv1_2, conv2_1_2])
    con_conv2_1 = ReLU()(con_conv2_1)


    #layer2_2
    conv2_2_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_1)
    conv2_2_1 = BatchNormalization()(conv2_2_1)
    conv2_2_1 = ReLU()(conv2_2_1)

    conv2_2_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_2_1)
    conv2_2_2 = BatchNormalization()(conv2_2_2)

    con_conv2_2 = Add()([con_conv2_1, conv2_2_2])
    con_conv2_2 = ReLU()(con_conv2_2)


    #layer2_3
    conv2_3_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_2)
    conv2_3_1 = BatchNormalization()(conv2_3_1)
    conv2_3_1 = ReLU()(conv2_3_1)


    conv2_3_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_3_1)
    conv2_3_2 = BatchNormalization()(conv2_3_2)

    con_conv2_3 = Add()([con_conv2_2, conv2_3_2])
    con_conv2_3 = ReLU()(con_conv2_3)            


    con_conv2_3_1 = Conv2D(32, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv2_3)  #240*320
    con_conv2_3_1 = BatchNormalization()(con_conv2_3_1)
    con_conv2_3_1 = ReLU()(con_conv2_3_1)


    #layer3_1
    conv3_1_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)  
    conv3_1_1 = BatchNormalization()(conv3_1_1)
    conv3_1_1 = ReLU()(conv3_1_1)
    pooling3_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv3_1_1)    


    conv3_1_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling3_1_1)
    conv3_1_2 = BatchNormalization()(conv3_1_2)

    con_conv3_1 = Add()([con_conv2_3_1,conv3_1_2])
    con_conv3_1 = ReLU()(con_conv3_1)

    #layer3_2
    conv3_2_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_1)
    conv3_2_1 = BatchNormalization()(conv3_2_1)
    conv3_2_1 = ReLU()(conv3_2_1)

    conv3_2_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_2_1)
    conv3_2_2  = BatchNormalization()(conv3_2_2)

    con_conv3_2 = Add()([con_conv3_1,conv3_2_2 ])
    con_conv3_2 = ReLU()(con_conv3_2)

    #layer3_3
    conv3_3_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_2)
    conv3_3_1 = BatchNormalization()(conv3_3_1)
    conv3_3_1 = ReLU()(conv3_3_1)


    conv3_3_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_3_1)
    conv3_3_2 = BatchNormalization()(conv3_3_2)

    con_conv3_3 = Add()([con_conv3_2,conv3_3_2])
    con_conv3_3 = ReLU()(con_conv3_3)    
    con_conv3_3_1 = Conv2D(64, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv3_3)  #240*320
    con_conv3_3_1 = BatchNormalization()(con_conv3_3_1)
    con_conv3_3_1 = ReLU()(con_conv3_3_1)


    #layer4_1
    conv4_1_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    conv4_1_1 = BatchNormalization()(conv4_1_1)
    conv4_1_1 = ReLU()(conv4_1_1)
    pooling4_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv4_1_1)  #120*160

    conv4_1_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling4_1_1)
    conv4_1_2 = BatchNormalization()(conv4_1_2)

    con_conv4_1 = Add()([con_conv3_3_1,conv4_1_2])
    con_conv4_1 = ReLU()(con_conv4_1)

    #layer4_2
    conv4_2_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv4_1)
    conv4_2_1 = BatchNormalization()(conv4_2_1)
    conv4_2_1 = ReLU()(conv4_2_1)

    conv4_2_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_2_1)
    conv4_2_2  = BatchNormalization()(conv4_2_2)

    con_conv4_2 = Add()([con_conv4_1,conv4_2_2 ])
    con_conv4_2 = ReLU()(con_conv4_2)



    #layer4_3
    conv4_3_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv4_2)
    conv4_3_1 = BatchNormalization()(conv4_3_1)
    conv4_3_1 = ReLU()(conv4_3_1)


    conv4_3_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_3_1)
    conv4_3_2 = BatchNormalization()(conv4_3_2)

    con_conv4_3 = Add()([con_conv4_2,conv4_3_2])      
    con_conv4_3 = ReLU()(con_conv4_3)                  



    conv_mlc4_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
    conv_mlc4_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc4_1)
    conv_mlc4_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_1)


    conv_mlc4_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
    conv_mlc4_3 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc4_3)
    conv_mlc4_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_3)

    conv_mlc4_5 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
    conv_mlc4_5 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc4_5)
    conv_mlc4_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_5)

#     conv_mlc4_7 = Conv2D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_7 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc4_7)
#     conv_mlc4_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_7)

#     conv_mlc4_9 = Conv2D(64, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_9 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc4_9)
#     conv_mlc4_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_9)

#     conv_mlc4_11 = Conv2D(64, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_11 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc4_11)
#     conv_mlc4_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_11)

    con_mlc4 = Concatenate(axis=-1)([conv_mlc4_1,conv_mlc4_3,conv_mlc4_5])

    con_mlc4 = Lambda(lambda x: K.mean(x,-1))(con_mlc4)





    conv_mlc3_1 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    conv_mlc3_1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc3_1)
    conv_mlc3_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_1)


    conv_mlc3_3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    conv_mlc3_3 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc3_3)
    conv_mlc3_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_3)

    conv_mlc3_5 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    conv_mlc3_5 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc3_5)
    conv_mlc3_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_5)

#     conv_mlc3_7 = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_7 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc3_7)
#     conv_mlc3_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_7)
#     
#     conv_mlc3_9 = Conv2D(32, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_9 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc3_9)
#     conv_mlc3_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_9)

#     conv_mlc3_11 = Conv2D(32, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_11 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc3_11)
#     conv_mlc3_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_11)

    con_mlc3 = Concatenate(axis=-1)([conv_mlc3_1,conv_mlc3_3,conv_mlc3_5])

    con_mlc3 = Lambda(lambda x: K.mean(x,-1))(con_mlc3)



    conv_mlc2_1 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
    conv_mlc2_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc2_1)
    conv_mlc2_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_1)


    conv_mlc2_3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
    conv_mlc2_3 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc2_3)
    conv_mlc2_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_3)

    conv_mlc2_5 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
    conv_mlc2_5 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc2_5)
    conv_mlc2_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_5)

#     conv_mlc2_7 = Conv2D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_7 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc2_7)
#     conv_mlc2_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_7)

#     conv_mlc2_9 = Conv2D(16, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_9 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc2_9)
#     conv_mlc2_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_9)

#     conv_mlc2_11 = Conv2D(16, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_11 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc2_11)
#     conv_mlc2_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_11)

    con_mlc2 = Concatenate(axis=-1)([conv_mlc2_1,conv_mlc2_3,conv_mlc2_5])
    con_mlc2 = Lambda(lambda x: K.mean(x,-1))(con_mlc2)



    up4 = UpSampling2D(size = (2,2),interpolation='bilinear')(con_mlc4)
    fpn3_conv_out = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_mlc3)
    fpn3_out = Add()([up4,fpn3_conv_out])


    up3 = UpSampling2D(size = (2,2),interpolation='bilinear')(fpn3_out)
    fpn2_conv_out = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_mlc2)
    fpn2_out = Add()([up3,fpn2_conv_out])


    conv_blam1_out = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fpn2_out)
    conv_blam1_out  = Conv2D(1, 1, activation = 'sigmoid')(conv_blam1_out)


    model = Model(input = inputs, output = conv_blam1_out)


    return model
