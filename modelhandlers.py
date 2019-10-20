from sklearn import svm, metrics
from datetime import datetime

import numpy as np
import os


logfile = None


def initLogfile():
    global logfile

    os.makedirs("logs", exist_ok=True)
    logfile = open(os.path.join("logs", "%s.log" % (str(datetime.now()))
            .replace(".", "-")
            .replace(":", "-")
            .replace(" ", "_")
        ), "w")


def log(text="", end="\n"):
    global logfile

    print(text, end=end)

    if logfile is not None:
        logfile.write(text + end)
        logfile.flush()


def eerCalc(yTrue, yPred):
    if type(yPred[0]) is not np.float64 and len(yPred) > 0 and len(yPred[0]) > 1:
        yPred = [p for n, p in yPred]
    
    fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred, pos_label=1)
    fnr = 1 - tpr
    index = np.nanargmin(np.absolute((fnr - fpr)))

    return max(fpr[index], fnr[index]), thresholds[index]


def eerScore(yTrue, yPred):
    eer, _ = eerCalc(yTrue, yPred)
    return eer


def tuneModel(xTrain, yTrain, xTest, yTest, kernel='rbf', probability=True, loging=False):
    # TODO: Figure out this LinearSVC

    if loging:
        print("Initializing log file...")
        initLogfile()
    
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
