import random
import cnn_dataset_files

class data_ultils:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    num_types = 0
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test , self.num_types= cnn_dataset_files.load_datasets()
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)

    def get_data_batch(self, batch_size):
        x = []
        y = []
        train_len = len(self.y_train)
        select_list = random.sample(range(train_len), batch_size)
        for select in select_list:
            x.append(self.x_train[select])
            y.append(self.y_train[select])
        return x, y

    def get_data_lenght(self):
        return len(self.y_train)
    
    def get_num_types(self):
        return self.num_types