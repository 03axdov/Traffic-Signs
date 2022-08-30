import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dense

from keras.applications import MobileNetV2
from keras.models import load_model


def streetsignsModel(n_classes):
    
    input = Input(shape=(60,60,3))

    x = Conv2D(16, (3,3), activation="relu")(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=input, outputs=x)


def mobileNet(input_shape, n_classes):
    model = MobileNetV2(weights="imagenet", input_shape=input_shape, include_top=False)

    penultimate_layer = model.layers[-3]
    new_layer = GlobalAveragePooling2D()(penultimate_layer.output)
    new_output_layer = Dense(n_classes, activation="softmax")(new_layer)
    new_model = Model(inputs=model.input, outputs=new_output_layer)

    return new_model