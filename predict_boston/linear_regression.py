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
