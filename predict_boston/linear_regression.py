#import necessary packages
import numpy as np
import json 
#read training data
data_path = 'data/housing.data'
data = np.fromfile(data_path, sep=' ')
