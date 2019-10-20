from sklearn import svm, metrics
from datetime import datetime

import numpy as np
import os


logfile = None


def initLogfile():
    """Creates a new log file in the ./logs folder
    """

    global logfile

    os.makedirs("logs", exist_ok=True)
    logfile = open(os.path.join("logs", "%s.log" % (str(datetime.now()))
            .replace(".", "-")
            .replace(":", "-")
            .replace(" ", "_")
        ), "w")


def log(text="", end="\n"):
    """Prints text in the terminal and appends it to a log file if one
    was previoulsy created using initLogFile

    Args:
        text (str, optional): Text to be logged
        end (str, optional): The suffix of the logged text (simmilar to print's end parameter)
    """

    global logfile

    print(text, end=end)

    if logfile is not None:
        logfile.write(text + end)
        logfile.flush()


def eerCalc(yTrue, yPred):
    """Calculates the EER and the corresponding threshold using a ROC curve

    Args:
        yTrue (list(int)): Correct labels
        yPred (list(int) or list((float, int))): Simple or probabilistic classification results
    
    Returns:
        (float, float): The EER and the corresponding threshold
    """

    if type(yPred[0]) is not np.float64 and len(yPred) > 0 and len(yPred[0]) > 1:
        yPred = [p for n, p in yPred]
    
    fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred, pos_label=1)
    fnr = 1 - tpr
    index = np.nanargmin(np.absolute((fnr - fpr)))

    return max(fpr[index], fnr[index]), thresholds[index]


def eerScore(yTrue, yPred):
    """Calculates the EER using a ROC curve

    Args:
        yTrue (list(int)): Correct labels
        yPred (list(int) or list((float, int))): Simple or probabilistic classification results
    
    Returns:
        float: The EER
    """

    eer, _ = eerCalc(yTrue, yPred)
    return eer


def tuneModel(xTrain, yTrain, xTest, yTest, kernel='rbf', probability=True, loging=False):
    """Tunes an SVM model with given data

    Args:
        xTrain (list(list(int))): List of training histograms
        yTrain (list(int)): Correct labels for the training histograms
        xTest (list(list(int))): List of test histograms
        yTest (list(int)): Correct labels for the test histograms
        kernel (str, optional): The kernel of the SVM (can be 'rbf' or 'linear')
        probability (bool, optional): If set probabilistic SVM training will be used
        loging (bool, optional): If set tuning statistics will be saved in a log file

    Returns:
        (SVM, float, callable): The best SVM, its EER threshold and a predictor for later use
    """

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
    """Trains an SVM model with given data and parameters

    Args:
        xTrain (list(list(int))): List of training histograms
        yTrain (list(int)): Correct labels for the training histograms
        xTest (list(list(int))): List of test histograms
        yTest (list(int)): Correct labels for the test histograms
        C (float): The C parameter of the SMV
        gamma (float): The gamma parameter of the SMV
        kernel (str, optional): The kernel of the SVM (can be 'rbf' or 'linear')
        varbose (bool, optional): If set to true training information will be printed to the console

    Returns:
        (SVM, float, callable): The trained SVM, the EER threshold and a predictor for later use
    """

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
    """Classifies a list of feature vectors using a given SVM.

    By defalt the return value is a list of predicted labels but if the feature vectors
    are given along with a bounding box (result of face detection), that box will also be
    appended to the preducted label (as a tuple). If probability is set to True the probabilistic
    prediction result will also be appended.

    Args:
        clf (SVM): The classifier to be used
        X (list(list(int)) or list(list((int, (int, int, int, int))))): The list of feature vectors (histograms), optionally along with the detected face box
        threshold (float): The decision threshold
        yTrue (list(list(int)), optional): If given prediction statistics will be printed to the console
        probability (bool, optional): If true probabilistics results will be returned
    
    Returns:
        list(int) or list((int, (int, int, int, int))) or list((int, (int, int, int, int), float)): The calculated results (described above)
    """

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
