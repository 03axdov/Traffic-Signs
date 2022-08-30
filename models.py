import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dense


def streetsignsModel(n_classes):    # 8875 trainable parameters with 43 classes
    
    input = Input(shape=(60,60,3))  # 60 - Rough mean of images height and width

    x = Conv2D(15, (3,3), activation="relu")(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3,3), activation="relu")(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu")(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=input, outputs=x)