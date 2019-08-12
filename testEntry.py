from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np

from lbpcalc import LBPCalc
import data


def trainModel(X, yTrain, yTest, C, gamma):
    # print("Fitting the model...")
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma) # , verbose=True)
    clf.fit(X, yTrain)
    # print("Model fitted")

    # print("Testing...")
    yPred = clf.predict(X_test)
    conMat = metrics.confusion_matrix(yTest, yPred)
    far = conMat[1][0] / (conMat[1][0] + conMat[0][0])
    frr = conMat[0][1] / (conMat[0][1] + conMat[1][1])
    print("FAR = %f, FRR = %f" %(far, frr))


lbp = LBPCalc(((16, 2), (8, 2)))


def maattaHistogram(img):
    # Total histograms
    hist_16_2 = lbp.histogram(img, (16, 2))
    hist_8_2 = lbp.histogram(img, (8, 2))

    # 3x3 overlapping regions with a 14 pixel overlap
    localHists = []
    for xOffset in [-7, 14, 35]:
        for yOffset in [-7, 14, 35]:
            localHists += lbp.histogram(img, (8, 2), windowSize=(35, 35), xOffset=xOffset, yOffset=yOffset)
    
    return hist_16_2 + localHists + hist_8_2


# Generating the histograms and savaing them in files
print("Processing data...")
data.getTrainingData("NormalizedFace", "train", "out3", maattaHistogram)
data.getTrainingData("NormalizedFace", "test", "out3", maattaHistogram)
print("Data processed")

print("Loading data...")
X_train, y_train = data.getTrainingDataFromFile("out3", "train")
X_test, y_test = data.getTrainingDataFromFile("out3", "test")
print("Data loaded")

# TODO: Try to fit this
trainModel(X_train, y_train, y_test, C=1e6, gamma=8e-12) # FAR = 0.29, FRR = 0.009
trainModel(X_train, y_train, y_test, C=1e10, gamma=8e-12) # FAR = 0.31, FRR = 0.009
trainModel(X_train, y_train, y_test, C=9e2, gamma=8e-12) # FAR = 0.37, FRR = 0.064
trainModel(X_train, y_train, y_test, C=3e4, gamma=4e-9) # FAR = 0.27, FRR = 0.053