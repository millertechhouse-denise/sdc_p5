import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import training as tr
import helper_functions as hp

import os
import os.path

cars = []
notcars = []
for dirpath, dirnames, filenames in os.walk("./non-vehicles"):
    for filename in [f for f in filenames if f.endswith(".png")]:
        print(os.path.join(dirpath, filename))
        notcars.append(os.path.join(dirpath, filename))
        
for dirpath, dirnames, filenames in os.walk("./vehicles"):
    for filename in [f for f in filenames if f.endswith(".png")]:
        print(os.path.join(dirpath, filename))
        cars.append(os.path.join(dirpath, filename))

# TODO play with these values to see how your classifier
# performs under different binning scenarios
#spatial = 50
#histbin = 24

#car_features = tr.extract_features(cars, cspace='YUV', spatial_size=(spatial, spatial),
#                        hist_bins=histbin, hist_range=(0, 256))
#notcar_features = tr.extract_features(notcars, cspace='YUV', spatial_size=(spatial, spatial),
#                        hist_bins=histbin, hist_range=(0, 256))

car_features = hp.extract_features(cars, color_space='RGB2YCrCb')
notcar_features = hp.extract_features(notcars, color_space='RGB2YCrCb')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

#print('Using spatial binning of:',spatial,
 #   'and', histbin,'histogram bins')
#print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
joblib.dump(svc, 'saved_svc.pickle') 
joblib.dump(X_scaler, 'saved_scalar.pickle') 
clf = joblib.load('saved_svc.pickle')
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
