from data_processing import train_validation_split, process_test, data_generators
from models import streetsignsModel, mobileNet

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    if False:
        train_path = '' # Location of the dataset
        new_train_path = '' # Location of the processed training split of the dataset
        new_validation_path = '' # Location of the processed validtation split of the dataset
        train_validation_split(train_path=train_path, new_train_path=new_train_path, new_validation_path=new_validation_path)

        process_test(path='C:\\Users\\axeld\\Downloads\\GTSRB\\Test', csv_path='C:\\Users\\axeld\\Downloads\\GTSRB\\Test.csv')

    BATCH_SIZE = 64
    EPOCHS = 1

    train_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\training_data\\train'
    validation_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\training_data\\validation'
    test_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\Test'

    train_generator, validation_generator, test_generator = data_generators(BATCH_SIZE, train_path, validation_path, test_path)

    model_path = './Models'
    model_saver = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10
    )

    n_classes = train_generator.num_classes
    model = mobileNet(input_shape=(60,60,3), n_classes=n_classes)
    model.summary()

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=["accuracy"])

    model.fit(train_generator,
            validation_data=validation_generator, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            callbacks=[model_saver, early_stopping])
        
    
if __name__ == '__main__':
    main()