from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from methods import maattaHistogram, colorspaceHistogram, grayscaleLBP, curiousMethod
from datetime import datetime

import numpy as np
import argparse
import os
import cv2

import data


# TODO: Get access to databeses
# TODO: Train and test the HSV/YCrCb method
# TODO: Figure out a third method


def eerCalc(yTrue, yPred):
    if type(yPred[0]) is not np.float64 and len(yPred) > 0 and len(yPred[0]) > 1:
        yPred = [p for n, p in yPred]
    
    fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred, pos_label=1)
    fnr = 1 - tpr
    index = np.nanargmin(np.absolute((fnr - fpr)))

    return fpr[index], thresholds[index]


def eerScore(yTrue, yPred):
    eer, _ = eerCalc(yTrue, yPred)
    return eer


def tuneModel(xTrain, yTrain, xTest, yTest, kernel='rbf', probability=True):
    # TODO: Figure out this LinearSVC
    
    gammas = [a * 10 ** exp for exp in range(-6, -12, -1) for a in range(1, 10, 2)]
    Cs = [a * 10 ** exp for exp in range(0, 10) for a in range(1, 10, 2)]

    bestEer = 100.0
    bestThreshold = 0.0
    bestCfl = None
    bestGamma = 0.0
    bestC = 0.0

    for C in Cs:
        for gamma in gammas:
            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, tol=gamma, probability=probability)

            if kernel == "linear":
                clf = svm.LinearSVC(C=C, tol=gamma, fit_intercept=True, max_iter=100000)

            clf.fit(xTrain, yTrain)

            yTrue = yTest
            yPred = []
            if kernel == "linear":
                yPred = clf.decision_function(xTest)
            else:
                yPred = clf.predict_proba(xTest) if probability else clf.predict(xTest)

            eer, threshold = eerCalc(yTrue, yPred)
            log("EER = %f for C = %e, gamma = %e" % (eer, C, gamma), end=" ")


            if eer < bestEer and eer > 0.00001:
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

    return bestCfl, bestThreshold, lambda x: (bestCfl.predict_proba([x])[:,1] >= bestThreshold).astype(bool)[0]


def trainModel(xTrain, yTrain, xTest, yTest, C, gamma, kernel='rbf', verbose=False):
    if verbose:
        print("Fitting the model...")
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True) # , verbose=verbose)
    clf.fit(xTrain, yTrain)
    if verbose:
        print("Model fitted")

    if verbose:
        print("Testing...")
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

    return clf, threshold, lambda x: (clf.predict_proba([x])[:,1] >= threshold).astype(bool)[0]


def predict(clf, X, threshold, yTrue=None, probability=False):
    if len(X) > 0 and len(X[0]) > 1:
        xTest = [x[0] for x in X]
        box = [x[1] for x in X]
    else:
        xTest = X

    yPredProb = clf.predict_proba(xTest)
    yPred = (yPredProb[:,1] >= threshold).astype(bool)

    if yTrue:
        conMat = metrics.confusion_matrix(yTrue, yPred)
        frr = conMat[0][1] / (conMat[0][1] + conMat[0][0])
        far = conMat[1][0] / (conMat[1][0] + conMat[1][1])

        print("Confusion matrix:")
        print(conMat)
        print()
        print("FAR =", far)
        print("FRR =", frr)

    result = [yPred]
    if len(X) > 0 and len(X[0]) > 1:
        result.append(box)
    if probability:
        result.append([p[1 if p[1] >= threshold else 0] for p in yPredProb])

    return list(zip(*result))


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
parser.add_argument("--method", "-me", choices=["Maatta", "HSV", "YCrCb", "Gray", "Dual", "Curious"])

parser.add_argument("--dataPath", "-dpa")
parser.add_argument("--dataPrefix", "-dpr")

parser.add_argument("--trainTrue", "-trt")
parser.add_argument("--trainSpoof", "-trs")
parser.add_argument("--testTrue", "-tet")
parser.add_argument("--testSpoof", "-tes")

parser.add_argument("--modelPath", "-mp")
parser.add_argument("--photoPath", "-pp")

parser.add_argument("--kernel", "-k", choices=["rbf", "linear"], default="rbf")
parser.add_argument("--C", "-c")
parser.add_argument("--Gamma", "-g")

parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--log", "-l", action="store_true")

args = parser.parse_args()

mode = args.mode
method = args.method


if args.log:
    os.makedirs("logs", exist_ok=True)
    logfile = open(os.path.join("logs", "%s.log" % (str(datetime.now()))
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
    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    methodDict = {
        "Maatta" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, maattaHistogram, grayscale=True
        ),
        "HSV" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="HSV"), grayscale=False
        ),
        "YCrCb" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="YCrCb"), grayscale=False
        ),
        "Dual" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="Dual"), grayscale=False
        ),
        "Gray" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, grayscaleLBP, grayscale=True
        ),
        "Curious" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, curiousMethod, grayscale=False
        )
    }

    if args.method is not None:
        if args.testTrue is None and args.testSpoof is None:
            prin("Processing data...")
            methodDict[args.method]("all", args.trainTrue, args.trainSpoof)
            print("Data processed")
        elif args.testTrue is None or args.testSpoof is None:
            print("Error: Only one test set given")
        else:
            print("Processing data...")
            methodDict[args.method]("train", args.trainTrue, args.trainSpoof)
            methodDict[args.method]("test", args.testTrue, args.testSpoof)
            print("Data processed")
    else:
        print("Error: Method not given")
elif mode == "Tune":
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

    clf, threshold, _ = tuneModel(xTrain, yTrain, xTest, yTest, kernel=args.kernel)

    if args.modelPath is not None:
        if args.method is None:
            print("Error: Method not given")
        else:
            print("Saving model to file...")
            data.saveClassifier(clf, threshold, args.method, args.modelPath)    
            print("Model saved")
elif mode == "Train":
    # TODO: Loading with 4-fold CV (if _train/_test files don't exist, check _all)

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

    clf, threshold, spoofPredictor = trainModel(xTrain, yTrain, xTest, yTest, C=C, gamma=gamma, kernel=args.kernel, verbose=args.verbose)

    if args.modelPath is not None:
        if args.method is None:
            print("Error: Method not given")
        else:
            print("Saving model to file...")
            data.saveClassifier(clf, threshold, args.method, args.modelPath)    
            print("Model saved")
elif mode == "Class":
    methodDict = {
        "Maatta" : (maattaHistogram, True),
        "HSV" : (lambda img: colorspaceHistogram(img, space="HSV"), False),
        "YCrCb" : (lambda img: colorspaceHistogram(img, space="YCrCb"), False),
        "Dual" : (lambda img: colorspaceHistogram(img, space="Dual"), False),
        "Gray" : (grayscaleLBP, True),
        "Curious" : (curiousMethod, False)
    }

    dataPath = args.dataPath
    modelPath = args.modelPath

    if dataPath is None:
        print("Error: Data path not given")

    if modelPath is None:
        print("Error: Model path not given")

    clf, threshold, method = data.loadClassifier(modelPath)
    X = data.loadPhoto(dataPath, methodDict[method][0], methodDict[method][1])
    y = predict(clf, X, threshold, probability=True)

    print(y)

    # This will be removed soon:
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(dataPath)
    for yy in y:
        c = (0, 0, 255) if yy[0] else (0, 255, 0)
        x, y, w, h = yy[1]
        
        text = "Imposter" if yy[0] else "Client"
        text += " %f" % (yy[2] * 100) + "%"

        cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
        cv2.putText(img, text, (x, y + h + 15), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + h + 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Result", img)
    cv2.waitKey()

# python .\testEntry.py -mo Gen -me Gray -dpa "out/gray/" -dpr "from_raw" -trt "raw/client_train_raw.txt" -trs "raw/imposter_train_raw.txt" -tet "raw/client_test_raw.txt" -tes "raw/imposter_test_raw.txt"
# python .\testEntry.py -mo Tune -k rbf -dpa "out/gray/" -dpr "from_raw" -l
# python .\testEntry.py -mo Train -k rbf -dpa "out/gray/" -dpr "from_raw" -c "3" -g "5e-9" -v
