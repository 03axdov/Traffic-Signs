import os
import glob
import random
import shutil
import numpy as np
import pandas as pd

import tensorflow as tf

def split_dataset(ds, train_split=0.9, shuffle=True):
    if shuffle:
       random.shuffle(ds)
    
    train_size = int(len(ds) * train_split)
    
    train_ds = ds[0:train_size]
    val_ds = ds[train_size+1:]
    
    return train_ds, val_ds

def train_validation_split(train_path, new_train_path, new_validation_path):
    folders = os.listdir(train_path)
    for t, folder in enumerate(folders):
        full_path = os.path.join(train_path, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))  # All images are PNGs

        x_train, x_val = split_dataset(images_paths)

        print(f"FOLDER: {t}")
        print(f"x_train: {np.shape(x_train)}")
        print(f"x_val: {np.shape(x_val)}")

        for x in x_train:
            path_to_folder = os.path.join(new_train_path, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(new_validation_path, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)


def process_test(path, csv_path):

    csv = pd.read_csv(csv_path)
    for i in range(len(csv['ClassId'])):

        img_name = csv['Path'][i].replace('Test/', '')
        label = csv['ClassId'][i]

        path_to_folder = os.path.join(path, str(label))

        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)

        img_path = os.path.join(path, img_name)
        shutil.move(img_path, path_to_folder)


def data_generators(BATCH_SIZE, train_path, validation_path, test_path):
    
    preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
        rescalue= 1 / 255
    )

    train_generator = preprocessor.flow_from_directory(
        train_path,
        target_size=(60,60),
        color_mode='rgb',
        class_mode='sparse',   # Not One-Hot Encoded
        batch_size=BATCH_SIZE
    )

    validation_generator = preprocessor.flow_from_directory(
        validation_path,
        target_size=(60,60),
        color_mode='rgb',
        class_mode='sparse',   # Not One-Hot Encoded
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_generator = preprocessor.flow_from_directory(
        test_path,
        target_size=(60,60),
        color_mode='rgb',
        class_mode='sparse',   # Not One-Hot Encoded
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator