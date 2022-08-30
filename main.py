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

    TRAIN = False
    TEST = True

    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001

    train_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\training_data\\train'
    validation_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\training_data\\validation'
    test_path = 'C:\\Users\\axeld\\Downloads\\GTSRB\\Test'

    train_generator, validation_generator, test_generator = data_generators(BATCH_SIZE, train_path, validation_path, test_path)


    if TRAIN:

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
        # model = mobileNet(input_shape=(60,60,3), n_classes=n_classes)
        model = streetsignsModel(n_classes=n_classes)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"])

        model.fit(train_generator,
                validation_data=validation_generator, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                callbacks=[model_saver, early_stopping])
    
    elif TEST:
        model = tf.keras.models.load_model("./Models")
        model.summary()

        print("TEST SET:")
        model.evaluate(test_generator)

    
if __name__ == '__main__':
    main()