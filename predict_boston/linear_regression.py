import numpy as np
import json 

def load_data():
    #read training data
    data_path = 'data/housing.data'
    data = np.fromfile(data_path, sep=' ')
    #reshape data into a 2D array
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    #training set ratio
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset, :]
    
    #normalize the data
    data_mean = np.mean(training_data, axis=0)
    data_max = np.max(training_data, axis=0)
    data_min = np.min(training_data, axis=0)
    for i in range(feature_num):
        data[:, i] = (data[:, i] - data_min[i]) / (data_max[i] - data_min[i])

    training_data = data[:offset, :]
    testing_data = data[offset:, :]
    return training_data, testing_data
    
    #get training and testing data
    training_data, testing_data = load_data()
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
