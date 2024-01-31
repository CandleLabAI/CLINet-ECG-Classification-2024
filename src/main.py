# ------------------------------------------------------------------------
# Copyright (c) 2024 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objs as go
import plotly.io as pio

from iccad_dataloader import DataGenerator
from mitbih_dataloader import load_mitbih_dataset
from tsne import tSNE
from involution import Involution
from network import create_lstm_inv_model

def iccad():
    # Load CSV files
    train = pd.read_csv('/data/iccad/data-indices/train_indice.csv')
    test = pd.read_csv('/data/iccad/data-indices/test_indice.csv')

    # Define classes and initial count
    classes = ['AFb', 'AFt', 'SR', 'SVT', 'VFb', 'VFt', 'VPD', 'VT']
    initialCount = [1074, 343, 12673, 944, 1263, 266, 953, 12697]

    # Initialize data structures for train and test data
    train = {'filename':[], 'label':[]}
    test = {'filename':[], 'label':[]}
    count_train = [0] * len(classes)
    count_test = [0] * len(classes)
    limitT = 0.8
    limitT1 = 0.2

    # Populate train and test data
    for filename in os.listdir('/data/iccad/tinyml_contest_data_training'):
        cat = filename.split('-')[1]

        for i, class_name in enumerate(classes):
            if cat == class_name and count_train[i] < initial_count[i] * limit_train:
                train_data['filename'].append(filename)
                train_data['label'].append(cat)
                count_train[i] += 1
            elif cat == class_name and count_test[i] < initial_count[i] * limit_test:
                test_data['filename'].append(filename)
                test_data['label'].append(cat)
                count_test[i] += 1
        
    # Create DataFrames from the train and test data dictionaries
    trainData = pd.DataFrame.from_dict(train)
    testData = pd.DataFrame.from_dict(test)

    # SAVE CSVs
    os.mkdir('data/iccad/data_indices1/')
    trainData.to_csv('data/iccad/data_indices1/train_indice.csv', index=False)
    testData.to_csv('data/iccad/data_indices1/test_indice.csv', index=False)

    path_data = '/data/iccad/tinyml_contest_data_training'
    path_indices = '/data/iccad/data_indices1'
    SIZE = 1250

    # Train Data
    train_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.shuffle(10).batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    one_element = train_iterator.get_next()
    x, y = one_element[...,0:-1], one_element[...,-1]
    x = np.expand_dims(x, axis=2)

    # Test Data
    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
    x_test = np.expand_dims(x_test, axis=2)

    xTrain = x
    yTrain = y
    xTest = x_test
    yTest = y_test

    # Create Conv-LSTM-Inv model
    model = create_lstm_inv_model(num_classes = 8)
    model.compile(optimizer=Adam(learning_rate=0.001),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=5, verbose=1),
    keras.callbacks.ModelCheckpoint(check_pt, monitor = 'val_accuracy', mode="max", verbose = 1, save_best_only = True, save_weights_only = True)
    ]

    history = model.fit(
            xTrain,
            yTrain,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_data=(xTest, yTest),
            callbacks=callbacks
            )

    # model.save('saved_model/conv_lstm_inv_model')
    # ## zip the saved model to make it downloadable
    # !zip -r 313641-50epochs.zip /saved_model/conv_lstm_inv_model
    # ## download the zip file from link
    # from IPython.display import FileLink
    # FileLink(r'313641-50epochs.zip')

    # Model Evaluation
    pred = model.predict(xTest).argmax(axis=1)
    print(classification_report(yTest.numpy(), pred, digits=4))
    cm = confusion_matrix(yTest, pred.tolist())
    res = []
    for c in range(8):
        tp = cm[c,c]
        fp = sum(cm[:,c]) - cm[c,c]
        fn = sum(cm[c,:]) - cm[c,c]
        tn = sum(np.delete(sum(cm)-cm[c,:],c))

        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        specificity = tn/(tn+fp)
        f1_score = 2*((precision*recall)/(precision+recall))
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        npv = tn/(tn+fn)
        
        res.append([round(precision,4),round(recall,4),round(specificity,4),round(npv,4), round(accuracy,4)])

    print(pd.DataFrame(res,columns = ['Precision','Recall','Specificity','N   P   V','Accuracy']).to_string(index=False))

    # Plot tSNE
    classes = ['AFb', 'AFt', 'SR', 'SVT', 'VFb', 'VFt', 'VPD', 'VT']
    fig = tSNE(lstm_inv_model, xTest, yTest, '(CNN+LSTM+INV) Model', classes)
    fig.show()

    # Save plot as PDF
    # fig.write_image('tsne-iccad.pdf', format='pdf') 


def mitbih():
    path = '/data/mit-bih/mitbih_database/'

    # Load the dataset
    X, y = load_mitbih_dataset(dataset_path)

    # Appending the type index in X itself
    for i in range(0, len(X)):
        X[i] = np.append(X[i], y[i])
    trainData = pd.DataFrame(X)

    # Train and Test Data
    trainData, testData = train_test_split(trainData, test_size=0.20)

    trainLabels = trainData[trainData.shape[1] - 1]
    testLabels = testData[testData.shape[1] - 1]
    trainData = trainData.iloc[:, :trainData.shape[1] - 1].values
    testData = testData.iloc[:, :testData.shape[1] - 1].values
    trainData = trainData.reshape(len(trainData), trainData.shape[1], 1)
    testData = testData.reshape(len(testData), testData.shape[1], 1)

    # Create Conv-LSTM-Inv model
    model = create_lstm_inv_model(num_classes = 5)

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=5, verbose=1),
    keras.callbacks.ModelCheckpoint(check_pt, monitor = 'val_accuracy', mode="max", verbose = 1, save_best_only = True, save_weights_only = True)
    ]

    history = model.fit(
            trainData,
            trainLabels,
            epochs=50, 
            batch_size=32,
            shuffle=True,
            validation_data=(testData, testLabels),
            callbacks=callbacks
            )

    # Model Evaluation
    pred = model.predict(testData).argmax(axis=1)
    print(classification_report(testLabels, pred, digits=4))

    cm = sklearn.metrics.confusion_matrix(testLabels, pred)
    res = []
    for c in range(5):
        tp = cm[c,c]
        fp = sum(cm[:,c]) - cm[c,c]
        fn = sum(cm[c,:]) - cm[c,c]
        tn = sum(np.delete(sum(cm)-cm[c,:],c))

        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        specificity = tn/(tn+fp)
        f1_score = 2*((precision*recall)/(precision+recall))
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        npv = tn/(tn+fn)

        res.append([round(precision,4),round(recall,4),round(specificity,4),round(npv,4), round(accuracy,4)])

    print(pd.DataFrame(res,columns = ['Precision','Recall','Specificity','N     P     V','Accuracy']).to_string(index=False))

    # Plot tSNE
    classes=['N', 'S', 'V', 'F', 'Q']
    fig = tSNE(lstm_inv_model, testData, testLabels, '(CNN+LSTM+INV) Model', classes)
    fig.show()

    # Save plot as PDF
    # fig.write_image('tsne-mitbih.pdf', format='pdf') 

if __name__ == "__main__":
    iccad()
    mitbih()