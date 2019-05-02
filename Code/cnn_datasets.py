import random
import json
import cnn_file_ultil

class data_ultils:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    num_types = 0
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test , self.num_types= cnn_file_ultil.load_datasets()
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)
        # show id and lable
        with open(cnn_file_ultil.lable_file_name) as json_file:    
            dict_lables = json.load(json_file)
            print("data_ultils -------------")
            print("dict_lables ", dict_lables)
            print("data_ultils -------------")

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