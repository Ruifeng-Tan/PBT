import numpy as np
import random
import os
import json
random.seed(2024)
np.random.seed(2024)

def split_func(data_path, life_label, dataset):
    # Here is the train_ids provided by BatteryML

    tmp_files = os.listdir(data_path)
    life_labels = json.load(open(life_label))
    files = [i for i in tmp_files if i in life_labels]

    test_ratio = 0.2
    testing_set = random.sample(files, int(len(files)*0.2))
    training_set = [i for i in files if i not in testing_set]

    val_ratio = 0.2

    validation_set= random.sample(training_set, int(len(training_set)*0.25))
    training_set = [i for i in training_set if i not in validation_set]


    assert len(training_set) + len(validation_set) + len(testing_set) == len(files)

    print(f'{dataset} spliting results: ')
    print(f'training_set: {len(training_set)} {training_set}')
    print(f'Val set: {len(validation_set)} {validation_set}')
    print(f'Test set: {len(testing_set)} {testing_set}')

data_path = './processed/ZNcoin'
life_label = '../Life labels/ZN-coin_labels.json'
dataset = 'ZNcoin'
split_func(data_path, life_label, dataset)
