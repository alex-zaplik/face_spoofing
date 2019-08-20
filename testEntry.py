from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from methods import maattaHistogram
from datetime import datetime

import numpy as np
import argparse

import data


# TODO: Get access to databeses
# TODO: Train and test the HSV/YCrCb method
# TODO: Figure out a third method


def eerCalc(yTrue, yPred):
    if len(yPred) > 0 and len(yPred[0]) > 1:
        yPred = [p for n, p in yPred]
    
    fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred, pos_label=1)
    fnr = 1 - tpr
    index = np.nanargmin(np.absolute((fnr - fpr)))

    return fpr[index], thresholds[index]


def eerScore(yTrue, yPred):
    eer, _ = eerCalc(yTrue, yPred)
    return eer


def tuneModel(xTrain, yTrain, xTest, yTest, probability=True):
    gammas = [a * 10 ** exp for exp in range(-6, -12, -1) for a in range(1, 10, 2)]
    Cs = [a * 10 ** exp for exp in range(0, 10) for a in range(1, 10, 2)]

    bestEer = 100.0
    bestThreshold = 0.0
    bestCfl = None
    bestGamma = 0.0
    bestC = 0.0

    for C in Cs:
        for gamma in gammas:
            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=probability)
            clf.fit(xTrain, yTrain)
            yTrue, yPred = yTest, clf.predict_proba(xTest) if probability else clf.predict(xTest)
            eer, threshold = eerCalc(yTrue, yPred)
            log("EER = %f for C = %e, gamma = %e" % (eer, C, gamma), end=" ")


            if eer < bestEer:
                log("<- New Best")

                bestEer = eer
                bestThreshold = threshold
                bestCfl = clf
                bestGamma = gamma
                bestC = C
            else:
                log()
    
    log()
    log("Best parameters found for:\n\tC = %e, gamma = %e\n\tEER = %f" % (bestC, bestGamma, bestEer))

    return bestCfl, lambda x: (bestCfl.predict_proba([x])[:,1] >= bestThreshold).astype(bool)[0]


def trainModel(xTrain, yTrain, xTest, yTest, C, gamma, verbose=False):
    # print("Fitting the model...")
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True) # , verbose=True)
    clf.fit(xTrain, yTrain)
    # print("Model fitted")

    # print("Testing...")
    yPredProb = clf.predict_proba(xTest)
    eer, threshold = eerCalc(yTest, yPredProb)
    yPred = (yPredProb[:,1] >= threshold).astype(bool)
    conMat = metrics.confusion_matrix(yTest, yPred)

    if verbose:
        print("Parameters: C = %e, gamma = %e, threshold = %f" %(C, gamma, threshold))
        print()
        print(conMat)
        print()
        print("EER = %f" % (eer))
        print()
        print(metrics.classification_report(yTest, yPred))
        print()

    return clf, lambda x: (clf.predict_proba([x])[:,1] >= threshold).astype(bool)[0]


# trainModel(X_train, y_train, X_test, y_test, C=3e6, gamma=1e-11, probability=True) # EER = 0.084474
# trainModel(X_train, y_train, X_test, y_test, C=7e2, gamma=3e-10, probability=True) # EER = 0.073171
# clf, spoofPredictor = trainModel(X_train, y_train, X_test, y_test, C=3e3, gamma=7e-11) # EER = 0.069007

# Modes:
# - Gen -   Generates new trainting data
#           Args: method, dataPath, dataPrefix, trainTrue, trainSpoof [, testTrue, testSpoof]
# - Tune -  Tunes a model on the specified data
#           Args: method, dataPath, dataPrefix, modelPath
# - Train - Tunes a model on the specified data with given parameters
#           Args: method, dataPath, dataPrefix, modelPath, C, gamma
# - Class - Performs classification with a model and a photo
#           Args: modelPath, photoPath

parser = argparse.ArgumentParser()

parser.add_argument("--mode", "-mo", default="Class", choices=["Gen", "Tune", "Train", "Class"])
parser.add_argument("--method", "-me")

parser.add_argument("--dataPath", "-dpa")
parser.add_argument("--dataPrefix", "-dpr")

parser.add_argument("--trainTrue", "-trt")
parser.add_argument("--trainSpoof", "-trs")
parser.add_argument("--testTrue", "-tet")
parser.add_argument("--testSpoof", "-tes")

parser.add_argument("--modelPath", "-mp")
parser.add_argument("--photoPath", "-pp")

parser.add_argument("--C", "-c")
parser.add_argument("--Gamma", "-g")

parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--log", "-l", action="store_true")

args = parser.parse_args()

mode = args.mode
method = args.method


if args.log:
    logfile = open("%s.log" % (str(datetime.now())
            .replace(".", "-")
            .replace(":", "-")
            .replace(" ", "_")
        ), "w")


def log(text="", end="\n"):
    print(text, end=end)

    if args.log:
        logfile.write(text + end)
        logfile.flush()


if mode == "Gen":
    # TODO: Use method

    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    if args.testTrue is None and args.testSpoof is None:
        print("Processing data...")
        data.getTrainingData(dataPath, dataPrefix, "all", args.trainTrue, args.trainSpoof, maattaHistogram, grayscale=True)
        print("Data processed")
    elif args.testTrue is None or args.testSpoof is None:
        print("Error: Only one test set given")
    else:
        print("Processing data...")
        data.getTrainingData(dataPath, dataPrefix, "train", args.trainTrue, args.trainSpoof, maattaHistogram, grayscale=True)
        data.getTrainingData(dataPath, dataPrefix, "test", args.testTrue, args.testSpoof, maattaHistogram, grayscale=True)
        print("Data processed")
elif mode == "Tune":
    # TODO: Use method
    # TODO: Save clf to file (modelPath)
    # TODO: Loading with 4-fold CV (if _train/_test files don't exist, check _all)

    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    if dataPath is None or dataPrefix is None:
        missing = [i[1] for i in [(dataPath, "dataPath"), (dataPrefix, "dataPrefix")] if i[0] is None]
        print("Error: Not all data path parameters given. Misssing", missing)

    print("Loading data...")
    xTrain, yTrain = data.getTrainingDataFromFile(dataPath, dataPrefix, "train")
    xTest, yTest = data.getTrainingDataFromFile(dataPath, dataPrefix, "test")
    print("Data loaded")

    clf = tuneModel(xTrain, yTrain, xTest, yTest)
elif mode == "Train":
    # TODO: Use method
    # TODO: Save clf to file (modelPath)
    # TODO: Loading with 4-fold CV

    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    C = eval(args.C)
    gamma = eval(args.Gamma)

    if dataPath is None or dataPrefix is None:
        missing = [i[1] for i in [(dataPath, "dataPath"), (dataPrefix, "dataPrefix")] if i[0] is None]
        print("Error: Not all data path parameters given. Misssing", missing)

    if C is None or gamma is None:
        missing = [i[1] for i in [(C, "C"), (gamma, "Gamma")] if i[0] is None]
        print("Error: Not all model parameters given. Misssing", missing)

    print("Loading data...")
    xTrain, yTrain = data.getTrainingDataFromFile(dataPath, dataPrefix, "train")
    xTest, yTest = data.getTrainingDataFromFile(dataPath, dataPrefix, "test")
    print("Data loaded")

    clf, spoofPredictor = trainModel(xTrain, yTrain, xTest, yTest, C=C, gamma=gamma, verbose=args.verbose)
elif mode == "Class":
    # TODO: Implements single image classification
    print("TODO: Implements single image classification")

# python .\testEntry.py -mo Gen -dpa "out/maatta/" -dpr "from_raw" -trt "raw/client_train_raw.txt" -trs "raw/imposter_train_raw.txt" -tet "raw/client_test_raw.txt" -tes "raw/imposter_test_raw.txt"
# python .\testEntry.py -mo Tune -dpa "out/maatta/" -dpr "from_raw" -l
# python .\testEntry.py -mo Train -dpa "out/maatta/" -dpr "from_raw" -c "3e3" -g "7e-11" -v
