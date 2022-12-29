import os
import random

def make_filepath_list():
    train_file_list = []
    valid_file_list = []
    num_sumples = len(os.listdir('../dataset/train/images/off'))
    for top_dir in os.listdir('../dataset/train/images/'):
        file_dir = os.path.join('../dataset/train/images/',top_dir)
        if file_dir == '../dataset/train/images/._.DS_Store' or file_dir == '../dataset/train/images/.DS_Store':
            continue
        file_list = os.listdir(file_dir)
        random.shuffle(file_list)
        file_list = file_list[:num_sumples]

        

        #８割を学習データ、２割を検証データ
        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join('../dataset/train/images/',top_dir,file).replace('\\','/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join('../dataset/train/images/',top_dir,file).replace('\\','/') for file in file_list[num_split:]]

    return train_file_list, valid_file_list