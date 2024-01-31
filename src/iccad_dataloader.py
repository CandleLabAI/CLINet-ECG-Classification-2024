# ------------------------------------------------------------------------
# Copyright (c) 2024 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Data loader for training and testing
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

def loadCSV(csvf):
    """
    Load CSV file and return a dictionary containing information from the CSV.

    :param csv_file: CSV file name
    :return: {label: [file1, file2, ...]}
    """
    dict_labels = {}
    with open(csv_file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)  # skip header
        for row in csv_reader:
            filename, label = row[0], row[1]

            # append filename to current label
            if label in dict_labels:
                dict_labels[label].append(filename)
            else:
                dict_labels[label] = [filename]

    return dict_labels

def txt_to_numpy(filename, row):
    """
    Read a text file and convert it to a numpy array.

    :param filename: Name of the text file
    :param row: Number of rows in the numpy array
    :return: Numpy array
    """
    file = open(filename)
    lines = file.readlines()
    data_mat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        data_mat[row_count] = line[0]
        row_count += 1
        
    return data_mat

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_dir, indice_dir, mode, size):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []

        classes = ['AFb', 'AFt', 'SR', 'SVT', 'VFb', 'VFt', 'VPD', 'VT']
        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            cat = classes.index(str(k))
            for filename in v:     
                self.names_list.append(str(filename) + ' ' + str(cat))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + "/" + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(text_path):
          print(text_path + 'does not exist')
          print(self.names_list[idx].split(' ')[0])
          return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = np.append(IEGM_seg, label)
        return sample