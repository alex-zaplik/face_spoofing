from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from methods import maattaHistogram
from datetime import datetime

import numpy as np

import data


logfile = open("%s.log" % (str(datetime.now())
        .replace(".", "-")
        .replace(":", "-")
        .replace(" ", "_")
    ), "w")


def log(text="", end="\n"):
    print(text, end=end)
    logfile.write(text + end)
    logfile.flush()


def eerScore(yTrue, yPred):
    if len(yPred) > 0 and len(yPred[0]) > 1:
        yPred = [p for n, p in yPred]
    
    fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred, pos_label=1)
    fnr = 1 - tpr

    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]


def tuneModel(xTrain, yTrain, xTest, yTest, probability=True):
    # gammas = [a * 10 ** exp for exp in range(-6, -12, -1) for a in range(1, 10, 2)]
    # Cs = [a * 10 ** exp for exp in range(0, 10) for a in range(1, 10, 2)]

    gammas = [a * 10 ** exp for exp in range(-8, -11, -1) for a in range(1, 10, 2)]
    Cs = [a * 10 ** exp for exp in range(1, 6) for a in range(1, 10, 2)]

    bestEer = 100.0
    bestCfl = None
    bestGamma = 0.0
    bestC = 0.0

    for C in Cs:
        for gamma in gammas:
            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=probability)
            clf.fit(xTrain, yTrain)
            yTrue, yPred = yTest, clf.predict_proba(xTest) if probability else clf.predict(xTest)
            eer = eerScore(yTrue, yPred)
            log("EER = %f for C = %e, gamma = %e" % (eer, C, gamma), end=" ")


            if eer < bestEer:
                log("<- New Best")

                bestEer = eer
                bestCfl = clf
                bestGamma = gamma
                bestC = C
            else:
                log()
    
    log()
    log("Best parameters found for:\n\tC = %e, gamma = %e\n\tEER = %f" % (bestC, bestGamma, bestEer))

    return bestCfl


def trainModel(xTrain, yTrain, xTest, yTest, C, gamma, probability):
    # print("Fitting the model...")
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=probability) # , verbose=True)
    clf.fit(xTrain, yTrain)
    # print("Model fitted")

    # print("Testing...")
    yPred = []
    if probability:
        yPred = clf.predict_proba(xTest)
    else:
        yPred = clf.predict(xTest)

    # y_pred = (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool) # set threshold as 0.3
    print("EER = %f" % (eerScore(yTest, yPred)))

    return clf


# Generating the histograms and savaing them in files
# print("Processing data...")
# data.getTrainingData("NormalizedFace", "train", "out3", maattaHistogram)
# data.getTrainingData("NormalizedFace", "test", "out3", maattaHistogram)
# print("Data processed")

print("Loading data...")
X_train, y_train = data.getTrainingDataFromFile("out3", "train")
X_test, y_test = data.getTrainingDataFromFile("out3", "test")
print("Data loaded")

# TODO: Try to fit this
# tuneModel(X_train, y_train, X_test, y_test)

# EER = 0.091927 for C = 3.000000e+06, gamma = 1.000000e-11

# Best parameters found for:
#         C = 3.000000e+06, gamma = 1.000000e-11
#         EER = 0.091927

trainModel(X_train, y_train, X_test, y_test, C=3e6, gamma=1e-11, probability=True) # EER = 0.084474
trainModel(X_train, y_train, X_test, y_test, C=7e2, gamma=3e-10, probability=True) # EER = 0.073171
