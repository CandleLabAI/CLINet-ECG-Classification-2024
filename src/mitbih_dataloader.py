import os
import csv

def load_mitbih_dataset(path):
    files = next(os.walk(path))[2]
    files.sort()

    records = [os.path.join(path, name) for name in files if name.endswith('.csv')]
    annotations = [os.path.join(path, name) for name in files if not name.endswith('.csv')]

    window_size = 625

    annots_list = ['N', 'L', 'R', 'e', 'j', 'S', 'A', 'a', 'J', 'V', 'E', 'F', '/', 'f', 'Q']
    N_class = ['N', 'L', 'R', 'e', 'j']
    S_class = ['A', 'a', 'J', 'S']
    V_class = ['V', 'E']
    F_class = ['F']
    Q_class = ['/', 'f', 'Q']

    classes_no = [0, 0, 0, 0, 0]
    classes = [N_class, S_class, V_class, F_class, Q_class]

    X = []
    y = []

    for record in range(1, len(records)):
        signals = []
        with open(records[record], 'rt') as file:
            read = csv.reader(file, delimiter=',', quotechar='|')
            row_index = -1
            for row in read:
                if row_index >= 0:
                    signals.insert(row_index, int(row[1]))
                row_index += 1

        with open(annotations[record], 'r') as annoFile:
            data = annoFile.readlines()
            for d in range(1, len(data)):
                splitted = list(filter(None, data[d].split(' ')))
                pos = int(splitted[1])
                arrhythmia_type = splitted[2]

                for i, arrhythmia_class in enumerate(classes):
                    if arrhythmia_type in annots_list and arrhythmia_type in arrhythmia_class:
                        arrhythmia_index = i
                        classes_no[arrhythmia_index] += 1
                        if window_size <= pos < len(signals) - window_size:
                            beat = signals[pos - window_size: pos + window_size]
                            X.append(beat)
                            y.append(arrhythmia_index)

    return X, y