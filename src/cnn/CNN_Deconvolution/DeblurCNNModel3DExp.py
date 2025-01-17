import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW

# Class witch provides CNN training and giving answers
class DeblurCNNModel3DExp:
    def __init__(self):
        return

    def ModelBuilder(input_shape = (64, 64, 40, 1), learning_rate = 0.001):
        # Original source with tutorial: https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
        
        enc_channels_1 = 24
        enc_channels_2 = enc_channels_1 * 2
        enc_channels_3 = enc_channels_1 * 4

        enc_channels_4 = enc_channels_1 * 8

        # Define the input as a tensor with shape input_shape
        X_input = layers.Input(input_shape, name="img")

        # Stage 0: lvl-up channels
        X = layers.Conv3D(enc_channels_1, (1, 1, 1), name = 'conv0')(X_input)
        X = layers.BatchNormalization()(X)
        stageZeroOutput = layers.LeakyReLU()(X)

        # Stage 1: first Residual layer with (3x3) kernels and (40, 64, 64, *) input_shape
        X = layers.Conv3D(enc_channels_1, (3, 3, 3), name = 'conv1_1', padding='same')(stageZeroOutput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_1, (5, 5, 5), name = 'conv1_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_1, (3, 3, 3), name = 'conv1_3', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageZeroOutput])
        
        StageOneLongcut = layers.LeakyReLU()(X)
        stageOneOutput = layers.MaxPooling3D((2, 2, 2))(StageOneLongcut)
        
        # Stage 2: second Residual layer with (3x3) kernels and (32, 32, 1) input_shape
        X = layers.Conv3D(enc_channels_2, (1, 1, 1), name = 'conv2_0', padding='same')(stageOneOutput)
        X = layers.BatchNormalization()(X)
        stageTwoInput = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv2_1', padding='same')(stageTwoInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv2_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv2_3', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageTwoInput])
        
        StageTwoLongcut = layers.LeakyReLU()(X)
        stageTwoOutput = layers.MaxPooling3D((2, 2, 2))(StageTwoLongcut)

        # Stage 3: third Residual layer with (3x3) kernels and (16, 16, 1) input_shape
        X = layers.Conv3D(enc_channels_3, (1, 1, 1), name = 'conv3_0', padding='same')(stageTwoOutput)
        X = layers.BatchNormalization()(X)
        stageThreeInput = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv3_1', padding='same')(stageThreeInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv3_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv3_3', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageThreeInput])
        
        StageThreeLongcut = layers.LeakyReLU()(X)
        stageThreeOutput = layers.MaxPooling3D((2, 2, 2))(StageThreeLongcut)

        # Stage 4: fourth Residual layer with (5x5) kernels and (16, 16, 1) input_shape
        X = layers.Conv3D(enc_channels_4, (1, 1, 1), name = 'conv4_0', padding='same')(stageThreeOutput)
        X = layers.BatchNormalization()(X)
        stageFourInput = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_4, (3, 3, 3), name = 'conv4_1', padding='same')(stageFourInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_4, (3, 3, 3), name = 'conv4_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        X = layers.Conv3D(enc_channels_4, (3, 3, 3), name = 'conv4_3', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageFourInput])

        stageFourOutput = layers.LeakyReLU()(X)

        # Stage 5: upsampling info from 4th stage
        # Stage 5.1: prepare data from stage 1 and add two layers
        StageThreeLongcut = layers.Conv3D(enc_channels_3, (1, 1, 1), name = 'conv5_1', padding='same')(StageThreeLongcut)
        StageThreeLongcut = layers.BatchNormalization()(StageThreeLongcut)
        StageThreeLongcut = layers.LeakyReLU()(StageThreeLongcut)

        X = layers.UpSampling3D((2, 2, 2))(stageFourOutput)
        X = layers.Conv3D(enc_channels_3, (1, 1, 1), name = 'conv5_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Add()([StageThreeLongcut, X])
        stageFiveInput = layers.LeakyReLU()(X)
        
        # Stage 5.2: fourth Residual layer with (3x3) kernels and (20, 20, 1) input_shape
        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv5_6', padding='same')(stageFiveInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv5_7', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_3, (3, 3, 3), name = 'conv5_8', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageFiveInput])
        
        stageFiveOutput = layers.LeakyReLU()(X)

        # Stage 6: upsampling info from 4th stage
        # Stage 6.1: prepare data from stage 1 and add two layers
        StageTwoLongcut = layers.Conv3D(enc_channels_2, (1, 1, 1), name = 'conv6_1', padding='same')(StageTwoLongcut)
        StageTwoLongcut = layers.BatchNormalization()(StageTwoLongcut)
        StageTwoLongcut = layers.LeakyReLU()(StageTwoLongcut)

        X = layers.UpSampling3D((2, 2, 2))(stageFiveOutput)
        X = layers.Conv3D(enc_channels_2, (1, 1, 1), name = 'conv6_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Add()([StageTwoLongcut, X])
        stageSixInput = layers.LeakyReLU()(X)
        
        # Stage 6.2: fourth Residual layer with (3x3) kernels and (20, 20, 1) input_shape
        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv6_6', padding='same')(stageSixInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv6_7', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_2, (3, 3, 3), name = 'conv6_8', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageSixInput])
        
        stageSixOutput = layers.LeakyReLU()(X)

        # Stage 7: upsampling info from 2rd stage
        # Stage 7.1: prepare data from stage 2 and add two layers
        StageOneLongcut = layers.Conv3D(enc_channels_1, (1, 1, 1), name = 'conv7_1', padding='same')(StageOneLongcut)
        StageOneLongcut = layers.BatchNormalization()(StageOneLongcut)
        StageOneLongcut = layers.LeakyReLU()(StageOneLongcut)

        X = layers.UpSampling3D((2, 2, 2))(stageSixOutput)
        X = layers.Conv3D(enc_channels_1, (1, 1, 1), name = 'conv7_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Add()([StageOneLongcut, X])
        stageSevenInput = layers.LeakyReLU()(X)
        
        # Stage 7.2: fourth Residual layer with (3x3) kernels and (20, 20, 1) input_shape
        X = layers.Conv3D(enc_channels_1, (3, 3, 3), name = 'conv7_6', padding='same')(stageSevenInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_1, (5, 5, 5), name = 'conv7_7', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(enc_channels_1, (3, 3, 3), name = 'conv7_8', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageSevenInput])
        
        stageSevenOutput = layers.LeakyReLU()(X)

        # Stage 8: Stack outputs from 3, 4 stages
        # Stage 8.1: Prepair output from 3 stage [convert (*, 10, 10, 32) into (*, 20, 20, 32)]
        Z = layers.UpSampling3D((4, 4, 4))(stageFourOutput)
        Z = layers.Conv3D(enc_channels_2, (1, 1, 1), name = 'conv8_1', padding='same')(Z)
        Z = layers.BatchNormalization()(Z)
        Z = layers.LeakyReLU()(Z)

        X = layers.UpSampling3D((2, 2, 2))(stageFiveOutput)
        X = layers.Conv3D(enc_channels_2, (1, 1, 1), name = 'conv8_2', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        # Stage 8.2: Concatinate outputs from 3 and 4 stages [3rd (*, 20, 20, 32) and 4th (*, 20, 20, 32) -> (*, 20, 20, 64)]
        stageEightThreeStack = layers.Concatenate(axis=-1)([X, Z, stageSixOutput])
        
        # Stage 8.3: fifth Residual layer with (3x3) kernels and (20, 20, 1) input_shape
        X = layers.Conv3D(3*enc_channels_2, (3, 3, 3), name = 'conv8_4', padding='same')(stageEightThreeStack)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(3*enc_channels_2, (3, 3, 3), name = 'conv8_5', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(3*enc_channels_2, (3, 3, 3), name = 'conv8_6', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageEightThreeStack])
        
        stageEightOutput = layers.LeakyReLU()(X)

        # Stage 9: Stack outputs from 6 and 5 stages
        # Stage 9.1: Prepair output from 6 stage [convert (*, 32, 32, 64) into (*, 64, 64, 64)]
        X = layers.UpSampling3D((2, 2, 2))(stageEightOutput)
        X = layers.Conv3D(3 * enc_channels_1, (1, 1, 1), name = 'conv9_0', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)

        # Stage 9.2: Prepair output from 5 stage [convert (*, 64, 64, 32) into (*, 64, 64, 64)]
        Y = layers.Conv3D(3 * enc_channels_1, (1, 1, 1), name = 'conv9_1', padding='same')(stageSevenOutput)
        Y = layers.BatchNormalization()(Y)
        Y = layers.LeakyReLU()(Y)

        # Stage 9.3: Sum outputs from 6 and 5 stages
        X = layers.Add()([X, Y])
        stageNineInput = layers.LeakyReLU()(X)
        
        # Stage 9.4: sixth Residual layer with (5x5) kernels and (*, 64, 64, 64) input_shape
        X = layers.Conv3D(3*enc_channels_1, (3, 3, 3), name = 'conv9_5', padding='same')(stageNineInput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(3*enc_channels_1, (5, 5, 5), name = 'conv9_6', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Conv3D(3*enc_channels_1, (3, 3, 3), name = 'conv9_7', padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Add()([X, stageNineInput])
        
        stageNineOutput = layers.LeakyReLU()(X)

        # Stage 9: go down channels
        X = layers.Conv3D(1, (1, 1, 1), name = 'conv10', padding='same')(stageNineOutput)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        
        X = layers.Add()([X_input, X])
        X_output = layers.Activation('relu')(X)

        # Create model
        initial_learning_rate = learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=4380, decay_rate=0.95, staircase=True)

        model = keras.Model(inputs = X_input, outputs = X_output, name='DeblurCNNExp3D')
        model.compile(optimizer= AdamW(learning_rate=lr_schedule, beta_2 = 0.98, weight_decay=0.015),loss='mae')
        
        return model

