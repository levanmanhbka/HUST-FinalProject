import json
import cnn_file_ultil as file_handle

# Separate image
file_handle.make_dataset_folder()
file_handle.separate_sample()

 # show id and lable
with open(file_handle.lable_file_name) as json_file:    
    dict_lables = json.load(json_file)
    print("cnn_dataset_files -------------")
    print("dict_lables ", dict_lables)
    print("cnn_dataset_files -------------")