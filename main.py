from data_processing import train_validation_split, process_test

def main():
    if False:
        train_path = '' # Location of the dataset
        new_train_path = '' # Location of the processed training split of the dataset
        new_validation_path = '' # Location of the processed validtation split of the dataset
        train_validation_split(train_path=train_path, new_train_path=new_train_path, new_validation_path=new_validation_path)

        process_test(path='C:\\Users\\axeld\\Downloads\\GTSRB\\Test', csv_path='C:\\Users\\axeld\\Downloads\\GTSRB\\Test.csv')

    

if __name__ == '__main__':
    main()