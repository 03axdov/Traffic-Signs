import os
import glob
import random
import shutil

def split_dataset(ds, train_split=0.8, shuffle=True):
    if shuffle:
       random.shuffle(ds)
    
    train_size = len(ds)
    
    train_ds = ds[0:train_size]
    val_ds = ds[train_size+1:]
    
    return train_ds, val_ds

def train_validation_split(train_path, new_train_path, new_validation_path, validation_split=0.1):
    folders = os.listdir(train_path)
    for folder in folders:
        full_path = os.path.join(train_path, folder)
        images_paths = glob.glob(os.path.join(full_path), '*.png')  # All images are PNGs
        print(images_paths.shape)

        x_train, x_val = split_dataset(images_paths)
        print(x_train.shape)
        print(x_val.shape)

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