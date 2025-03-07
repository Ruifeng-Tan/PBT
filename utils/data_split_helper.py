import random
import os

random.seed(2021)

# help to split train, vali, test set
class Dataset_split_helper():
    def __init__(self, root_path):
        '''
        init the Dataset_split_helper
        :param root_path:dataset loading root path
        '''
        self.root_path = root_path

        self.train_files, self.val_files, self.test_files = self.split_dataset()

    def split_dataset(self):
        '''
        split the dataset randomly and make sure the split result is the same at each time by setting random seed
        :return: train_files, val_files, test_files
        '''
        file_path = os.listdir(self.root_path)
        file_names = [i for i in file_path if i.endswith('.csv')]

        total_samples = len(file_names)
        train_size = int(0.6 * total_samples)
        test_size = int(0.2 * total_samples)

        train_files = random.sample(file_names, train_size)
        remaining_files = [f for f in file_names if f not in train_files]
        test_files = random.sample(remaining_files, test_size)
        val_files = [f for f in remaining_files if f not in test_files]

        return train_files, val_files, test_files

root_path = '../dataset/LFP/tagged/V1/'
split_helper = Dataset_split_helper(root_path)
train_files = split_helper.train_files
val_files = split_helper.val_files
test_files = split_helper.test_files

print(len(train_files), train_files)
print(len(val_files), val_files)
print(len(test_files), test_files)

